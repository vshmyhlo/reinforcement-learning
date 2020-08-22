from collections import namedtuple

import torch


class History(object):
    def __init__(self, keys):
        self.container = namedtuple('Container', keys)
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, **kwargs):
        self.buffer.append(self.container(**kwargs))

    def build_rollout(self):
        rollout = {}
        for step in self.buffer:
            for k in step._fields:
                if k not in rollout:
                    rollout[k] = []
                rollout[k].append(getattr(step, k))

        for k in rollout:
            rollout[k] = torch.stack(rollout[k], 1)

        return self.container(**rollout)
