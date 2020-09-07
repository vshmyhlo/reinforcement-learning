import numpy as np


class SampleAverageAgent(object):
    def __init__(self, num_actions, e):
        self.num_actions = num_actions
        self.e = e
        self.action_value = np.zeros(num_actions)
        self.action_count = np.zeros(num_actions)

    def act(self, state):
        if np.random.uniform() < self.e:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.action_value)

    def update(self, action, reward):
        self.action_count[action] += 1
        self.action_value[action] += 1 / self.action_count[action] * (reward - self.action_value[action])

    def __repr__(self):
        return '{}(num_actions={}, e={})' \
            .format(self.__class__.__name__, self.num_actions, self.e)


class ConstantStepAgent(object):
    def __init__(self, num_actions, e, alpha):
        self.num_actions = num_actions
        self.e = e
        self.alpha = alpha
        self.action_value = np.zeros(num_actions)

    def act(self, state):
        if np.random.uniform() < self.e:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.action_value)

    def update(self, action, reward):
        self.action_value[action] += self.alpha * (reward - self.action_value[action])

    def __repr__(self):
        return '{}(num_actions={}, e={}, alpha={})' \
            .format(self.__class__.__name__, self.num_actions, self.e, self.alpha)
