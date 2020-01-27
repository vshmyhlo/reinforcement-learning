import gym
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
        reward = torch.tensor(reward, device=self.device)
        done = torch.tensor(done, device=self.device)

        return state, reward, done, meta
