import gym
import torch


class Torch(gym.Wrapper):
    def __init__(self, env, dtype, device):
        super().__init__(env)

        self.dtype = dtype
        self.device = device

    def reset(self):
        state = self.env.reset()

        state = torch.tensor(state, dtype=self.dtype, device=self.device)

        return state

    def step(self, action):
        action = action.data.cpu().numpy()

        state, reward, done, meta = self.env.step(action)

        state = torch.tensor(state, dtype=self.dtype, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)

        return state, reward, done, meta
