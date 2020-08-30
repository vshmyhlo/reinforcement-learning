import gym
import numpy as np
import torch


class Torch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)

        self.device = device

    def reset(self):
        state = self.env.reset()

        state = torch.tensor(state, dtype=map_dtype(state.dtype), device=self.device)

        return state

    def step(self, action):
        action = action.data.cpu().numpy()

        state, reward, done, meta = self.env.step(action)

        state = torch.tensor(state, dtype=map_dtype(state.dtype), device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)

        return state, reward, done, meta


def map_dtype(dtype):
    if dtype == np.float64:
        return torch.float
    else:
        return None
