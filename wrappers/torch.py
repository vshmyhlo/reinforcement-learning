import gym
import torch


class Torch(gym.Wrapper):
    def __init__(self, env, device=torch.device("cpu")):
        super().__init__(env)
        self.device = device

    def reset(self):
        state = self.env.reset()
        state = torch.tensor(state, device=self.device)

        return state

    def step(self, action):
        action = action.data.cpu().numpy()
        state, reward, done, info = self.env.step(action)

        state = torch.tensor(state, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)

        return state, reward, done, info
