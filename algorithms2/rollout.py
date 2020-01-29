import torch


class Rollout(object):
    def __init__(self, states=None, actions=None, rewards=None, dones=None, state_prime=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.state_prime = state_prime

    @classmethod
    def build(cls, history, state_prime):
        states, actions, rewards, dones = zip(*history)

        states = torch.stack(states, 1)
        actions = torch.stack(actions, 1)
        rewards = torch.stack(rewards, 1)
        dones = torch.stack(dones, 1)

        return cls(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            state_prime=state_prime)
