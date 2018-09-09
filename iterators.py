import numpy as np


# TODO: env.reset(d)

class HorizonIterator(object):
    def __init__(self, env, state_to_action, horizon):
        self.env = env
        self.state_to_action = state_to_action
        self.horizon = horizon

    def iterate(self, s, num_steps):
        for _ in range(num_steps):
            history = []

            for _ in range(self.horizon):
                a = self.state_to_action(s)
                s_prime, r, d, _ = self.env.step(a)
                history.append((s, a, r, d))
                s = np.where(np.expand_dims(d, -1), self.env.reset(d), s_prime)

            batch = {}
            batch['states'], batch['actions'], batch['rewards'], batch['dones'] = self.build_batch(history)

            yield batch, s

    @staticmethod
    def build_batch(history):
        columns = zip(*history)

        return [np.array(col).swapaxes(0, 1) for col in columns]
