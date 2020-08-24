from collections import namedtuple

import torch


class History(object):
    def __init__(self, keys):
        self.transition = namedtuple('Transition', keys)
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, **kwargs):
        self.buffer.append(self.transition(**kwargs))

    def build_rollout(self):
        rollout = {}
        for step in self.buffer:
            for k in step._fields:
                if k not in rollout:
                    rollout[k] = []
                rollout[k].append(getattr(step, k))

        for k in rollout:
            rollout[k] = torch.stack(rollout[k], 1)

        return self.transition(**rollout)
