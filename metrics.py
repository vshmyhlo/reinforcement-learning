import abc
import numpy as np


class Metric(abc.ABC):
    def compute(self):
        ...

    def update(self, value):
        ...

    def reset(self):
        ...


class Mean(abc.ABC):
    def __init__(self):
        self.data = []

    def compute(self):
        return sum(self.data) / len(self.data)

    def update(self, value):
        self.data.extend(np.reshape(value, [-1]))

    def reset(self):
        self.data = []
