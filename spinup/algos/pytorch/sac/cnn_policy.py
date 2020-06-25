import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from spinup.algos.pytorch.sac.core import *


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

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
        robot_state_shape = int(obs[0, 3, 0, 0].cpu().numpy())
        robot_state = obs[:, 3, 0, 1: robot_state_shape + 1]
    elif len(obs.shape) == 3:
        img = obs[:3, :, :]
        robot_state_shape = int(obs[3, 0, 0].cpu().numpy())
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


class SquashedGaussianCNNActor(nn.Module):

    def __init__(self, cnn, cnn_output_shape, state_input_shape, act_dim, state_fc_size, cat_fc_size, activation, act_limit, init_=lambda _: _, init2_=lambda _: _):
        super().__init__()
        self.cnn = cnn
        # self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.actor_state = mlp([state_input_shape] + list(state_fc_size), activation, activation,
                               init_)
        self.actor_fuse = mlp([cnn_output_shape + state_fc_size[-1]] + [cat_fc_size], activation, activation, init_)
        self.mu_layer = init2_(nn.Linear(cat_fc_size, act_dim))
        self.log_std_layer = nn.Linear(cat_fc_size, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        img, robot_state = split_obs(obs)
        # net_out = self.net(obs)
        cnn_out = self.cnn(img)
        state_out = self.actor_state(robot_state)
        h1 = torch.cat((cnn_out, state_out), 1)
        h2 = self.actor_fuse(h1)
        mu = self.mu_layer(h2)
        log_std = self.log_std_layer(h2)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class CNNQFunction(nn.Module):

    def __init__(self, cnn, cnn_output_shape, state_input_shape, state_fc_size, cat_fc_size, act_dim, activation, init_=lambda _: _, init2_=lambda _: _):
        super().__init__()
        self.cnn = cnn
        # self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.critic_state = mlp([state_input_shape + act_dim] + list(state_fc_size), activation, activation,
                                init_)
        self.critic_fuse = mlp([cnn_output_shape + state_fc_size[-1]] + [cat_fc_size], activation,
                               activation, init_)
        self.q = nn.Linear(cat_fc_size, 1)

    def forward(self, obs, act):
        img, robot_state = split_obs(obs)
        # net_out = self.net(obs)
        cnn_out = self.cnn(img)
        state_out = self.critic_state(torch.cat([robot_state, act], dim=-1))
        h1 = torch.cat((cnn_out, state_out), 1)
        h2 = self.critic_fuse(h1)
        # q = self.q(torch.cat([obs, act], dim=-1))
        q = self.q(h2)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class CNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, state_input_shape=4,
                 cnn_fc_size=512, state_fc_size=(64, 64),
                 cat_fc_size=128, cnn_activation=nn.ReLU, fc_activation=nn.Tanh, share_cnn=True, init_=lambda _: _):
        super().__init__()

        cnn_input_shape = observation_space.shape[0] - 1
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # shared network
        # init_ = lambda m: init(m,
        #                        nn.init.orthogonal_,
        #                        lambda x: nn.init.constant_(x, 0),
        #                        nn.init.calculate_gain('relu'))
        pi_cnn = CNNSharedNet(cnn_input_shape, cnn_fc_size, cnn_activation, init_)
        if share_cnn:
            q_cnn = pi_cnn
        else:
            q_cnn = CNNSharedNet(cnn_input_shape, cnn_fc_size, cnn_activation, init_)
        # build policy and value functions
        self.pi = SquashedGaussianCNNActor(pi_cnn, cnn_fc_size, state_input_shape, act_dim, state_fc_size, cat_fc_size, fc_activation, act_limit, init_)
        self.q1 = CNNQFunction(q_cnn, cnn_fc_size, state_input_shape, state_fc_size, cat_fc_size, act_dim, fc_activation)
        self.q2 = CNNQFunction(q_cnn, cnn_fc_size, state_input_shape, state_fc_size, cat_fc_size, act_dim, fc_activation)

        self.apply(weight_init)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs.unsqueeze(0), deterministic, False)
            return a.squeeze().cpu().numpy()
