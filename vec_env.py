from enum import Enum
from multiprocessing import Pipe, Process

import numpy as np

Command = Enum('Command', [
    'RESET',
    'STEP',
    'RENDER',
    'SEED',
    'GET_META',
    'CLOSE',
])


def worker(env_fn, conn):
    env = env_fn()

    while True:
        command, *data = conn.recv()

        if command is Command.RESET:
            conn.send(env.reset())
        elif command is Command.STEP:
            action, = data
            state, reward, done, meta = env.step(action)
            if done:
                state = env.reset()
            conn.send((state, reward, done, meta))
        elif command is Command.RENDER:
            mode, = data
            conn.send(env.render(mode=mode))
        elif command is Command.SEED:
            seed, = data
            conn.send(env.seed(seed))
        elif command is Command.GET_META:
            conn.send((env.observation_space, env.action_space, env.reward_range, env.metadata))
        elif command is Command.CLOSE:
            conn.send(env.close())
            break
        else:
            raise AssertionError('invalid command {}'.format(command))


class VecEnv(object):
    def __init__(self, env_fns):
        self.conns, child_conns = zip(*[Pipe(duplex=True) for _ in range(len(env_fns))])
        self.processes = [
            Process(target=worker, args=(env_fn, child_conn))
            for env_fn, child_conn in zip(env_fns, child_conns)]

        for process in self.processes:
            process.start()

        self.conns[0].send((Command.GET_META,))
        self.observation_space, self.action_space, self.reward_range, self.metadata = \
            self.conns[0].recv()

    def reset(self):
        for conn in self.conns:
            conn.send((Command.RESET,))

        state = [conn.recv() for conn in self.conns]
        state = np.array(state)

        return state

    def step(self, action):
        assert len(action) == len(self.conns)

        for conn, action in zip(self.conns, action):
            conn.send((Command.STEP, action))

        state, reward, done, meta = zip(*[conn.recv() for conn in self.conns])

        state = np.array(state)
        reward = np.array(reward)
        done = np.array(done)

        return state, reward, done, meta

    def render(self, mode='human'):
        self.conns[0].send((Command.RENDER, mode))

        return self.conns[0].recv()

    def seed(self, seed):
        for i, conn in enumerate(self.conns):
            conn.send((Command.SEED, seed + i))

        for conn in self.conns:
            conn.recv()

    def close(self):
        for conn in self.conns:
            conn.send((Command.CLOSE,))

        for conn in self.conns:
            conn.recv()

        for process in self.processes:
            process.join()
