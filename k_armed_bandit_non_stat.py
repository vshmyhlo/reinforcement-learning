import os
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from envs import KArmedBandit
from k_armed_bandit_agents import SampleAverageAgent, ConstantStepAgent


def evaluate_run(build_agent, steps):
    env = KArmedBandit(10, walk=0.01)
    agent = build_agent(env.action_space.n)

    rewards = np.zeros(steps)
    for t in range(steps):
        s = env.reset()
        a = agent.act(s)
        _, r, d, _ = env.step(a)
        assert d
        agent.update(a, r)
        rewards[t] = r

    return rewards


def main():
    runs = 5000
    steps = 10000

    with Pool(os.cpu_count()) as pool:
        for build_agent in [
            partial(SampleAverageAgent, e=0.1),
            partial(ConstantStepAgent, e=0.1, alpha=0.1),
            partial(ConstantStepAgent, e=0.1, alpha=0.01),
        ]:
            label = str(build_agent(10))

            rewards = np.zeros(steps)
            f = partial(evaluate_run, build_agent=build_agent, steps=steps)
            tasks = [pool.apply_async(f) for _ in range(runs)]
            for n, t in enumerate(tqdm(tasks, desc=label)):
                r = t.get()
                rewards += 1 / (n + 1) * (r - rewards)

            plt.plot(rewards, label=label)

    plt.legend()
    plt.ylim(0, None)
    plt.show()


if __name__ == "__main__":
    main()
