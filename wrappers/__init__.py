import gym
import numpy as np

from wrappers.adj_max import AdjMax
from wrappers.batch import Batch
from wrappers.grid_world import GridWorld
from wrappers.multi_agent import MultiAgent
from wrappers.skip_obs import SkipObs
from wrappers.stack_obs import StackObs
from wrappers.tensorboard_batch_monitor import TensorboardBatchMonitor
from wrappers.torch import Torch


class RandomFirstReset(gym.Wrapper):
    def __init__(self, env, num_steps):
        super().__init__(env)
        self.num_steps = num_steps
        self.first_reset = True

    def reset(self, **kwargs):
        if self.first_reset:
            self.first_reset = False
            steps = np.random.randint(self.num_steps)
            print(f"random reset: {steps}/{self.num_steps}")

            obs = self.env.reset()
            for _ in range(steps):
                obs, _, done, _ = self.env.step(self.env.action_space.sample())
                if done:
                    obs = self.env.reset()
        else:
            obs = super().reset(**kwargs)

        return obs
