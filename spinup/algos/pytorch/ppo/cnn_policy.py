import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from spinup.algos.pytorch.ppo.core import *


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def nature_cnn(init_, input_shape, output_size, activation):
    return nn.Sequential(
            init_(nn.Conv2d(input_shape, 32, 8, stride=4)),
            activation(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            activation(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            activation(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, output_size)),
            activation()
    )


def split_obs(obs):
    if len(obs.shape) == 4:
        img = obs[:, :3, :, :]
        robot_state_shape = int(obs[0, 3, 0, 0].numpy())
        robot_state = obs[:, 3, 0, 1: robot_state_shape + 1]
    elif len(obs.shape) == 3:
        img = obs[:3, :, :]
        robot_state_shape = int(obs[3, 0, 0].numpy())
        robot_state = obs[3, 0, 1: robot_state_shape + 1]
    else:
        raise ValueError
    return img, robot_state


class CNNSharedNet(nn.Module):
    def __init__(self, cnn_input_shape, cnn_fc_size, cnn_activation, init_):
        super(CNNSharedNet, self).__init__()
        self.cnn = nature_cnn(init_, cnn_input_shape, cnn_fc_size, cnn_activation)

    def forward(self, x):
        return self.cnn(x)


class ImgStateGaussianActor(nn.Module):

    def __init__(self, cnn_output_shape, state_input_shape, act_dim, state_fc_size, cat_fc_size, activation, init_=lambda _: _):
        super().__init__()
        self.actor_state = mlp([state_input_shape]+ list(state_fc_size), activation, activation, init_)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([cnn_output_shape + state_fc_size[-1]] + [cat_fc_size] + [act_dim], activation, init_=init_)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def forward(self, cnn_output, state_input, act=None):
        h1 = self.actor_state(state_input)
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        h2 = torch.cat((cnn_output, h1), 1)
        pi = self._distribution(h2)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class ImgStateCritic(nn.Module):

    def __init__(self, cnn_output_shape, state_input_shape, state_fc_size, cat_fc_size, activation, init_=lambda _: _):
        super().__init__()
        self.critic_state = mlp([state_input_shape] + list(state_fc_size), activation, activation,
                                init_)
        self.v_net = mlp([cnn_output_shape + state_fc_size[-1]] + [cat_fc_size] + [1], activation, init_=init_)

    def forward(self, cnn_output, state_input):
        h1 = self.critic_state(state_input)
        h2 = torch.cat((cnn_output, h1), 1)
        return torch.squeeze(self.v_net(h2), -1) # Critical to ensure v has right shape.


class ImgStateActorCriticDictBox(nn.Module):
    def __init__(self, observation_space, action_space,
                 state_input_shape=4,
                 cnn_fc_size=512, state_fc_size=(64, 64),
                 cat_fc_size=128, cnn_activation=nn.ReLU, fc_activation=nn.Tanh):
        super().__init__()

        cnn_input_shape = observation_space.shape[0] - 1

        # shared network
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.cnn = CNNSharedNet(cnn_input_shape, cnn_fc_size, cnn_activation, init_)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                               np.sqrt(2))
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi_ = ImgStateGaussianActor(cnn_fc_size, state_input_shape, action_space.shape[0], state_fc_size, cat_fc_size, fc_activation, init_)
        elif isinstance(action_space, Discrete):
            raise NotImplementedError
            # self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        # build value function
        self.v_ = ImgStateCritic(cnn_fc_size, state_input_shape, state_fc_size, cat_fc_size, fc_activation, init_)

    def pi(self, obs, act=None):
        img, robot_state = split_obs(obs)
        conv_features = self.cnn(img)
        return self.pi_(conv_features, robot_state, act)

    def v(self, obs):
        img, robot_state = split_obs(obs)
        conv_features = self.cnn(img)
        return self.v_(conv_features, robot_state)

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            img, robot_state = split_obs(obs)
            conv_features = self.cnn(img.unsqueeze(0)).squeeze()
            pi, _ = self.pi_(conv_features.unsqueeze(0), robot_state.unsqueeze(0))
            if deterministic:
                a = pi.mean
            else:
                a = pi.sample()
            a = a.squeeze()
            logp_a = self.pi_._log_prob_from_distribution(pi, a)
            v = self.v_(conv_features.unsqueeze(0), robot_state.unsqueeze(0))
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs, deterministic=False):
        return self.step(obs, deterministic)[0]