import gym
import numpy as np
import torch


class Torch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)

        self.device = device

    def reset(self):
        state = self.env.reset()

        state = torch.tensor(state, device=self.device)

        return state

    def step(self, action):
        action = action.data.cpu().numpy()

        state, reward, done, meta = self.env.step(action)

        state = torch.tensor(state, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)

        return state, reward, done, meta


class Batch(gym.Wrapper):
    def reset(self, **kwargs):
        state = self.env.reset()

        state = np.expand_dims(np.array(state), 0)

        return state

    def step(self, action):
        action = np.squeeze(action, 0)

        state, reward, done, meta = self.env.step(action)

        state = np.expand_dims(np.array(state), 0)
        reward = np.expand_dims(np.array(reward), 0)
        done = np.expand_dims(np.array(done), 0)
        meta = [meta]

        return state, reward, done, meta
