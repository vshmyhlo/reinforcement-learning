import torch


class History(object):
    def __init__(self):
        self.history = {
            'states': None,
            'actions': None,
            'rewards': None,
            'dones': None,
            'hidden': None,
            'states_prime': None,
        }

        self.size = 0

    def __len__(self):
        return self.size

    def append(self, state=None, action=None, reward=None, done=None, hidden=None, state_prime=None):
        updates = [
            ('states', state),
            ('actions', action),
            ('rewards', reward),
            ('dones', done),
            ('hidden', hidden),
            ('states_prime', state_prime),
        ]

        for k, v in updates:
            if v is None:
                assert self.history[k] is None
                continue
            if self.history[k] is None:
                self.history[k] = []
            self.history[k].append(v)

        self.size += 1

    def build_rollout(self):
        history = {
            k: torch.stack(self.history[k], 1) if self.history[k] is not None else None
            for k in self.history
        }

        return Rollout(
            states=history['states'],
            actions=history['actions'],
            rewards=history['rewards'],
            dones=history['dones'],
            hidden=history['hidden'],
            states_prime=history['states_prime'])


class Rollout(object):
    def __init__(self, states=None, actions=None, rewards=None, dones=None, hidden=None, states_prime=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.hidden = hidden
        self.states_prime = states_prime
