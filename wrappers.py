import torch


class Torch(object):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        state = self.env.reset()

        state = torch.tensor(state, device=self.device)

        state = transform_state(state)

        return state

    def step(self, action):
        action = action.data.cpu().numpy()

        state, reward, done, meta = self.env.step(action)

        state = torch.tensor(state, device=self.device)
        reward = torch.tensor(reward, device=self.device)
        done = torch.tensor(done, device=self.device)

        state = transform_state(state)

        return state, reward, done, meta

    def render(self, mode=None):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


def transform_state(state):
    if state.dim() == 3:
        state = state.permute(2, 0, 1)

    if state.dtype == torch.uint8:
        state = state.float() / 255

    return state
