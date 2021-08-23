import math

import gym
import torch
from torch import nn as nn


class Agent(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        encoder,
        num_features=64,
        memory=True,
    ):
        super().__init__()

        self.num_features = num_features
        self.memory = memory

        if encoder == "minigrid":
            self.encoder = MiniGridEncoder(action_space)
        elif encoder == "discrete":
            self.encoder = DiscreteEncoder(observation_space, num_features)

        self.rnn = LSTMCell(num_features)

        self.dist = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, action_space.n),
        )
        self.value = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, 1),
        )

        self.apply(self.weight_init)

    def forward(self, obs, action, memory):
        emb = self.encoder(obs, action)

        emb, memory = self.rnn(emb, memory)

        dist = torch.distributions.Categorical(logits=self.dist(emb))
        value = self.value(emb).squeeze(1)

        if not self.memory:
            memory = self.reset_memory(memory, torch.ones(obs.size(0), dtype=torch.bool))

        return dist, value, memory

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def zero_memory(self, batch_size):
        return self.rnn.zero_memory(batch_size)

    def reset_memory(self, memory, done):
        return self.rnn.reset_memory(memory, done)

    def detach_memory(self, memory):
        return self.rnn.detach_memory(memory)


class LSTMCell(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features
        self.rnn = nn.LSTMCell(num_features, num_features)

    def forward(self, input, memory):
        memory = self.rnn(input, memory)
        input, _ = memory

        return input, memory

    def zero_memory(self, batch_size):
        zeros = torch.zeros(batch_size, self.num_features)
        state = (zeros, zeros)
        return state

    def reset_memory(self, memory, done):
        done = done.unsqueeze(1)
        memory = tuple(torch.where(done, torch.zeros_like(x), x) for x in memory)
        return memory

    def detach_memory(self, memory):
        return tuple(x.detach() for x in memory)


class DiscreteEncoder(nn.Module):
    def __init__(self, observation_space: gym.spaces.Discrete, num_features: int):
        super().__init__()

        self.emb = nn.Embedding(observation_space.n, num_features)

    def forward(self, obs, action):
        return self.emb(obs)


class MiniGridEncoder(nn.Module):
    def __init__(self, action_space):
        super().__init__()

        self.obs_embedding = nn.Sequential(
            nn.Conv2d(20, 16, (2, 2)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 2)),  # TODO: smoothes?
            #
            nn.Conv2d(16, 32, (2, 2)),
            nn.LeakyReLU(0.2),
            #
            nn.Conv2d(32, 64, (2, 2)),
            nn.LeakyReLU(0.2),
        )
        self.action_embedding = nn.Embedding(action_space.n, 64)

    def forward(self, obs, action):
        obs = obs.float().permute(0, 3, 1, 2)
        obs = self.obs_embedding(obs)
        obs = obs.view(obs.size(0), obs.size(1))
        action = self.action_embedding(action)
        emb = obs + action

        return emb
