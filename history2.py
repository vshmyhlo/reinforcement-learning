import numpy as np
import torch


class History(object):
    def __init__(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append_transition(self):
        transition = Transition()
        self.buffer.append(transition)
        return transition

    def build(self):
        rollout = {}
        for transition in self.buffer:
            if len(rollout) == 0:
                for k in transition.data:
                    rollout[k] = []
            for k in transition.data:
                rollout[k].append(transition.data[k])

        for k in rollout:
            rollout[k] = torch.stack(rollout[k], 1)

        return rollout


class Transition(object):
    def __init__(self):
        self.data = {}

    def record(self, **kwargs):
        for k in kwargs:
            if k in self.data:
                raise ValueError("{} is already recorded".format(k))
            self.data[k] = kwargs[k]
