import gym
import numpy as np
from gym.spaces.box import Box
from collections import deque
import torch
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
#
#
# class SubProcWrapper(gym.Wrapper):
#     def __init__(self, env_fn):
#         env = SubprocVecEnv([env_fn])
#         env.reward_range = None
#         super(SubProcWrapper, self).__init__(env)
#
#     def step(self, action):
#         action = np.expand_dims(action, axis=0)
#         o, r, d, i = self.env.step(action)
#         o['img'] = o['img'].squeeze()
#         o['robot_state'] = o['robot_state'].squeeze()
#         o['task_state'] = o['task_state'].squeeze()
#         r = r[0]
#         d = d[0]
#         i = i[0]
#         return o, r, d, i
#
#     def reset(self):
#         o = self.env.reset()
#         o['img'] = o['img'].squeeze()
#         o['robot_state'] = o['robot_state'].squeeze()
#         o['task_state'] = o['task_state'].squeeze()
#         return o


class PyTorchWrapper(gym.Wrapper):
    def __init__(self, env, device):
        """Return only every `skip`-th frame"""
        super(PyTorchWrapper, self).__init__(env)
        self.device = device

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = torch.Tensor([reward])
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, reward, done, info


class DictTransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(DictTransposeImage, self).__init__(env)
        img_obs_shape = self.observation_space.spaces['img'].shape
        self.observation_space.spaces['img'] = Box(
            self.observation_space.spaces['img'].low[0, 0, 0],
            self.observation_space.spaces['img'].high[0, 0, 0],
            [img_obs_shape[2], img_obs_shape[1], img_obs_shape[0]],
            dtype=self.observation_space.spaces['img'].dtype)

    def observation(self, observation):
        observation['img'] = observation['img'].transpose(2, 0, 1)
        return observation


class DictToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DictToBoxWrapper, self).__init__(env)
        img_obs_shape = self.observation_space.spaces['img'].shape
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(img_obs_shape[0]+1, *img_obs_shape[1:] ), dtype='float32')

    def dict_to_box(self, dict_obs):
        new_obs = np.zeros(self.observation_space.shape)
        img_obs = dict_obs['img'].astype('float32') / 255
        new_obs[:3, :, :] = img_obs
        new_obs[3, 0, 0] = dict_obs['robot_state'].shape[0]
        new_obs[3, 1, 0] = dict_obs['task_state'].shape[0]
        new_obs[3, 0, 1:dict_obs['robot_state'].shape[0] + 1] = dict_obs['robot_state']
        new_obs[3, 1, 1:dict_obs['task_state'].shape[0] + 1] = dict_obs['task_state']
        return new_obs

    def observation(self, observation):
        new_obs = self.dict_to_box(observation)
        return new_obs


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, num_updates, num_update_steps, tb_writer=None, desired_rew_region=(0.4, 0.6), incr=0.002):
        self.env = env
        super(CurriculumWrapper, self).__init__(env)
        self.num_updates = num_updates
        self.num_update_steps = num_update_steps
        self.step_counter = 0
        self.update_counter = 0
        self.difficulty_cur = 0
        self.difficulty_reg = 0
        self.curr_episode_rewards = deque(maxlen=20)
        self.reg_episode_rewards = deque(maxlen=20)
        self.curr_success = deque(maxlen=20)
        self.reg_success = deque(maxlen=20)
        self.desired_rew_region = desired_rew_region
        self.incr = incr
        self.tb_writer = tb_writer
        self.num_regular_resets = 0
        self.num_resets = 0
        self.current_episode_return = 0

    def update_difficulties(self):
        if len(self.curr_success) > 1:
            if np.mean(self.curr_success) > self.desired_rew_region[1]:
                self.difficulty_cur += self.incr
            elif np.mean(self.curr_success) < self.desired_rew_region[0]:
                self.difficulty_cur -= self.incr
            self.difficulty_cur = np.clip(self.difficulty_cur, 0, 1)
        if len(self.reg_success) > 1:
            if np.mean(self.reg_success) > self.desired_rew_region[1]:
                self.difficulty_reg += self.incr
            elif np.mean(self.reg_success) < self.desired_rew_region[0]:
                self.difficulty_reg -= self.incr
            self.difficulty_reg = np.clip(self.difficulty_reg, 0, 1)

    def create_data_dict(self):
        return {'update_step': self.update_counter,
                'num_updates': self.num_updates,
                'eprewmean': None,
                'curr_eprewmean': np.mean(self.curr_episode_rewards) if len(self.curr_episode_rewards) > 1 else 0,
                'eval_eprewmean': None,
                'reg_eprewmean': np.mean(self.reg_episode_rewards) if len(self.reg_episode_rewards) > 1 else 0,
                'curr_success_rate': np.mean(self.curr_success) if len(self.curr_success) > 1 else 0,
                'reg_success_rate': np.mean(self.reg_success) if len(self.reg_success) > 1 else 0,
                'eval_reg_eprewmean': None,
                'difficulty_cur': self.difficulty_cur,
                'difficulty_reg': self.difficulty_reg}

    def step(self, action):
        self.step_counter += 1
        if self.step_counter % self.num_update_steps == 0:
            self.update_counter += 1
            self.update_difficulties()
            if self.tb_writer is not None:
                self.write_tb_log()
            self.num_regular_resets = 0
            self.num_resets = 0
        obs, rew, done, info = self.env.step(action)
        self.current_episode_return += rew
        if done:
            if 'reset_info' in info.keys() and info['reset_info'] == 'curriculum':
                self.curr_episode_rewards.append(self.current_episode_return)
                self.curr_success.append(float(info['task_success']))
                self.num_resets += 1
            elif 'reset_info' in info.keys() and info['reset_info'] == 'regular':
                self.reg_episode_rewards.append(self.current_episode_return)
                self.reg_success.append(float(info['task_success']))
                self.num_resets += 1
                self.num_regular_resets += 1
        return obs, rew, done, info

    def reset(self):
        data = self.create_data_dict()
        obs = self.env.reset(data)
        self.current_episode_return = 0
        return obs

    def write_tb_log(self):
        self.tb_writer.add_scalar("curr_success_rate", np.mean(self.curr_success) if len(self.curr_success) else 0, self.step_counter)
        self.tb_writer.add_scalar("reg_success_rate", np.mean(self.reg_success) if len(self.reg_success) else 0, self.step_counter)
        self.tb_writer.add_scalar("difficulty_cur", self.difficulty_cur, self.step_counter)
        self.tb_writer.add_scalar("difficulty_reg", self.difficulty_reg, self.step_counter)

        if len(self.curr_episode_rewards) > 1:
            self.tb_writer.add_scalar("curr_eprewmean_steps", np.mean(self.curr_episode_rewards), self.step_counter)
            self.tb_writer.add_scalar("regular_resets_ratio", self.num_regular_resets / self.num_resets if self.num_resets > 0 else 0,self.step_counter)
        if len(self.reg_episode_rewards) > 1:
            self.tb_writer.add_scalar("reg_eprewmean_steps", np.mean(self.reg_episode_rewards), self.step_counter)
        self.tb_writer.flush()

if __name__ == "__main__":
    from gym_grasping.envs.grasping_env import GraspingEnv
    env_fn = lambda: gym.make('stackVel_acgd-v0')
    env = PyTorchWrapper(DictToBoxWrapper(DictTransposeImage(SubProcWrapper(env_fn))), 'cpu')
    print(env.step([0,0,0,0,1])[3])