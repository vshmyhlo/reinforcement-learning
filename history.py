import torch


class History(object):
    def __init__(self):
        self.history = {
            'states': None,
            'actions': None,
            'rewards': None,
            'dones': None,
        }

    def append(self, state=None, action=None, reward=None, done=None):
        updates = [('states', state), ('actions', action), ('rewards', reward), ('dones', done)]

        for k, v in updates:
            if v is None:
                assert self.history[k] is None
                continue
            if self.history[k] is None:
                self.history[k] = []
            self.history[k].append(v)

    def build_rollout(self, state_prime=None):
        history = {
            k: torch.stack(self.history[k], 1) if self.history[k] is not None else None
            for k in self.history
        }

        return Rollout(
            states=history['states'],
            actions=history['actions'],
            rewards=history['rewards'],
            dones=history['dones'],
            state_prime=state_prime)


class Rollout(object):
    def __init__(self, states=None, actions=None, rewards=None, dones=None, state_prime=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.state_prime = state_prime
