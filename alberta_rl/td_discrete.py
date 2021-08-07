from collections import defaultdict

import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def agent(o, env: gym.Env):
    return env.action_space.sample()


def main():

    env = gym.make("Blackjack-v0")

    value = defaultdict(lambda: 0)
    count = defaultdict(lambda: 0)

    for _ in tqdm(range(100000)):
        history = []
        o = env.reset()
        ret = 0

        while True:
            a = agent(o, env)
            o_p, r, d, _ = env.step(a)
            ret += r

            history.append(o)

            o = o_p

            if d:
                break

        for o in history:
            value[o] += ret
            count[o] += 1

    value = {o: value[o] / count[o] for o in value}
    # print(value)

    x, y = np.meshgrid(
        np.arange(env.observation_space[0].n),
        np.arange(env.observation_space[1].n),
    )
    space = np.stack([x, y], 2)
    space = space.reshape((space.shape[0] * space.shape[1], 2))
    z = np.array([value.get((o[0], o[1], False), 0) for o in space])
    z = z.reshape((x.shape[0], x.shape[1]))

    # plt.contourf(x, y, z, levels=[-1.5, -0.5, 0.5, 1.5])
    plt.contourf(x, y, z, vmin=-1, vmax=1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
