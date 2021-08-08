from collections import defaultdict

import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ALPHA = 0.02
GAMMA = 1.0
EPS = 0.1


def agent(o, q, env: gym.Env):
    n = env.action_space.n
    if np.random.uniform() > EPS:
        return np.argmax([q[(o, a)] for a in range(n)])
    else:
        return np.random.randint(n)


def expected_action_value(q, o, env: gym.Env):
    n = env.action_space.n
    a = np.argmax([q[(o, a)] for a in range(n)])

    return q[(o, a)] * (1 - EPS) + EPS * sum(q[(o, a2)] for a2 in range(n) if a2 != a) / (n - 1)


def main():
    env = gym.make("Blackjack-v0")
    v = defaultdict(lambda: 0)
    q = defaultdict(lambda: 0)
    gs = []

    for _ in tqdm(range(100000)):
        g = 0
        o = env.reset()
        a = agent(o, q, env)
        while True:
            o_p, r, d, _ = env.step(a)
            g += r
            a_p = agent(o_p, q, env)
            if d:
                v[o] += ALPHA * (r - v[o])
                q[(o, a)] += ALPHA * (r - q[(o, a)])
            else:
                v[o] += ALPHA * (r + GAMMA * v[o_p] - v[o])
                q[(o, a)] += ALPHA * (r + GAMMA * expected_action_value(q, o_p, env) - q[(o, a)])

            o, a = o_p, a_p
            if d:
                break

        gs.append(g)

    plt.plot(np.convolve(gs, np.ones(100) / 100, mode="valid"))
    plt.show()

    x, y = np.meshgrid(
        np.arange(env.observation_space[0].n),
        np.arange(env.observation_space[1].n),
    )
    space = np.stack([x, y], 2)
    space = space.reshape((space.shape[0] * space.shape[1], 2))
    z = np.array([v.get((o[0], o[1], False), 0) for o in space])
    z = z.reshape((x.shape[0], x.shape[1]))

    print(z.min(), z.max())

    # plt.contourf(x, y, z, levels=[-1.5, -0.5, 0.5, 1.5])
    plt.contourf(x, y, z, vmin=-1, vmax=1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
