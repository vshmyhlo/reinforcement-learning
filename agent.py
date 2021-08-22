import gym
import torch
from torch import nn as nn


class Agent(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        encoder,
    ):
        super().__init__()

        if encoder == "minigrid":
            self.encoder = MiniGridEncoder(action_space)
        elif encoder == "discrete":
            self.encoder = DiscreteEncoder(observation_space)

        self.rnn = nn.LSTMCell(64, 64)
        self.dist = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n),
        )
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.apply(self.weight_init)

    def forward(self, obs, action, memory):
        emb = self.encoder(obs, action)

        memory = self.rnn(emb, memory)
        emb, _ = memory

        dist = torch.distributions.Categorical(logits=self.dist(emb))
        value = self.value(emb).squeeze(1)

        return dist, value, memory

    def zero_memory(self, batch_size):
        zeros = torch.zeros(batch_size, 64)
        state = (zeros, zeros)
        return state

    def reset_memory(self, state, done):
        done = done.unsqueeze(1)
        state = tuple(torch.where(done, torch.zeros_like(x), x) for x in state)
        return state

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class DiscreteEncoder(nn.Module):
    def __init__(self, observation_space: gym.spaces.Discrete):
        super().__init__()

        self.emb = nn.Embedding(observation_space.n, 64)

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
