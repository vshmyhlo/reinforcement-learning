import itertools
# from multiprocessing.pool import ThreadPool as Pool
# from multiprocessing import Pool
from multiprocessing import Pipe, Process
import gym
from tqdm import tqdm
import numpy as np
from enum import Enum

# TODO: refactor

# def __enter__(self)
# def __exit__(self, exc_type, exc_value, traceback)

Command = Enum('Command', ['RESET', 'STEP', 'CLOSE', 'GET_SPACES'])


def env_step(a):
    env, action = a
    return env.step(action)


def worker(env_fn, conn):
    env = env_fn()

    while True:
        command, *data = conn.recv()

        if command is Command.RESET:
            conn.send(env.reset())
        elif command is Command.STEP:
            action, = data
            conn.send(env.step(action))
        elif command is Command.GET_SPACES:
            conn.send((env.observation_space, env.action_space))
        elif command is Command.CLOSE:
            break
        else:
            raise AssertionError('invalid command {}'.format(command))


class VecEnv(object):
    def __init__(self, env_fns):
        self._conns, child_conns = zip(*[Pipe() for _ in range(len(env_fns))])
        self._processes = [Process(target=worker, args=(env_fn, child_conn))
                           for env_fn, child_conn in zip(env_fns, child_conns)]

        for process in self._processes:
            process.start()

        self._conns[0].send((Command.GET_SPACES,))
        observation_space, action_space = self._conns[0].recv()

        self.observation_space = VecObservationSpace(size=len(env_fns), observation_space=observation_space)
        self.action_space = VecActionSpace(size=len(env_fns), action_space=action_space)

    def reset(self, dones=None):
        if dones is None:
            for conn in self._conns:
                conn.send((Command.RESET,))

            state = np.array([conn.recv() for conn in self._conns])
        else:
            assert len(dones) == len(self._conns)

            state = np.zeros(self.observation_space.shape)

            for i, (conn, done) in enumerate(zip(self._conns, dones)):
                if done:
                    conn.send((Command.RESET,))

            for i, (conn, done) in enumerate(zip(self._conns, dones)):
                if done:
                    state[i] = conn.recv()

        return state

    def step(self, actions):
        assert len(actions) == len(self._conns)

        for conn, action in zip(self._conns, actions):
            conn.send((Command.STEP, action))

        state, reward, done, meta = zip(*[conn.recv() for conn in self._conns])

        state = np.array(state)
        reward = np.array(reward)
        done = np.array(done)

        return state, reward, done, meta

    def close(self):
        for conn in self._conns:
            conn.send((Command.CLOSE,))

        for process in self._processes:
            process.join()


class VecActionSpace(object):
    def __init__(self, size, action_space):
        self._size = size
        self._action_space = action_space

        self.n = action_space.n

    def sample(self):
        return np.array([self._action_space.sample() for _ in range(self._size)])


class VecObservationSpace(object):
    def __init__(self, size, observation_space):
        self.shape = [size, *observation_space.shape]


def main():
    env = VecEnv([lambda: gym.make('LunarLander-v2') for _ in range(8)])

    try:
        s = env.reset()

        for _ in tqdm(itertools.count()):
            a = env.action_space.sample()
            s, r, d, _ = env.step(a)

            s = np.where(np.expand_dims(d, -1), env.reset(d), s)
    finally:
        env.close()


if __name__ == '__main__':
    main()
