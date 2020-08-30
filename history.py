import numpy as np
import torch


# TODO: check all transitions have same keys
class History(object):
    def __init__(self, limit=None):
        if limit is not None:
            assert limit > 0

        self.limit = limit
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append_transition(self):
        transition = Transition()
        self.buffer.append(transition)

        if self.limit is not None:
            self.buffer = self.buffer[-self.limit:]

        return transition

    def full_rollout(self):
        return build_rollout(self.buffer)

    def sample_rollout(self, size):
        assert 0 < size <= len(self.buffer)
        start = np.random.randint(0, len(self.buffer) - size + 1)
        buffer = self.buffer[start:start + size]
        assert len(buffer) == size

        return build_rollout(buffer)


class Rollout(object):
    def __init__(self, data):
        self.data = data

    def __getattr__(self, key):
        return self.data[key]


class Transition(object):
    def __init__(self):
        self.data = {}

    def record(self, **kwargs):
        for k in kwargs:
            if k in self.data:
                raise ValueError('{} is already recorder'.format(k))
            self.data[k] = kwargs[k]


def build_rollout(buffer):
    rollout = {}
    for transition in buffer:
        if len(rollout) == 0:
            for k in transition.data:
                rollout[k] = []

        for k in transition.data:
            rollout[k].append(transition.data[k])

    for k in rollout:
        rollout[k] = torch.stack(rollout[k], 1)

    rollout = Rollout(rollout)

    return rollout
