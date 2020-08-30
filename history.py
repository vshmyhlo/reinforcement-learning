import torch


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


class History(object):
    def __init__(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append_transition(self):
        transition = Transition()
        self.buffer.append(transition)

        return transition

    # TODO: check all transitions have saame keys
    def build_rollout(self):
        rollout = {}
        for transition in self.buffer:
            if len(rollout) == 0:
                for k in transition.data:
                    rollout[k] = []

            for k in transition.data:
                rollout[k].append(transition.data[k])

        for k in rollout:
            rollout[k] = torch.stack(rollout[k], 1)

        rollout = Rollout(rollout)

        return rollout
