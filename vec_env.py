from multiprocessing import Pipe, Process
import numpy as np
from enum import Enum

# TODO: refactor
# def __enter__(self)
# def __exit__(self, exc_type, exc_value, traceback)

Command = Enum('Command', ['RESET', 'STEP', 'CLOSE', 'GET_SPACES', 'SEED'])


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
            state, reward, done, meta = env.step(action)
            if done:
                state = env.reset()
            conn.send((state, reward, done, meta))
        elif command is Command.GET_SPACES:
            conn.send((env.observation_space, env.action_space))
        elif command is Command.SEED:
            seed, = data
            conn.send(env.seed(seed))
        elif command is Command.CLOSE:
            break
        else:
            raise AssertionError('invalid command {}'.format(command))


class VecEnv(object):
    def __init__(self, env_fns):
        self.conns, child_conns = zip(*[Pipe() for _ in range(len(env_fns))])
        self.processes = [
            Process(target=worker, args=(env_fn, child_conn))
            for env_fn, child_conn in zip(env_fns, child_conns)]

        for process in self.processes:
            process.start()

        self.conns[0].send((Command.GET_SPACES,))
        observation_space, action_space = self.conns[0].recv()

        self.observation_space = VecObservationSpace(observation_space=observation_space)
        self.action_space = VecActionSpace(size=len(env_fns), action_space=action_space)

    def reset(self):
        for conn in self.conns:
            conn.send((Command.RESET,))

        state = np.array([conn.recv() for conn in self.conns])

        return state

    def step(self, actions):
        assert len(actions) == len(self.conns)

        for conn, action in zip(self.conns, actions):
            conn.send((Command.STEP, action))

        state, reward, done, meta = zip(*[conn.recv() for conn in self.conns])

        state = np.array(state)
        reward = np.array(reward)
        done = np.array(done)

        return state, reward, done, meta

    def close(self):
        for conn in self.conns:
            conn.send((Command.CLOSE,))

        for process in self.processes:
            process.join()

    def seed(self, seed):
        for i, conn in enumerate(self.conns):
            conn.send((Command.SEED, seed**i))

        for conn in self.conns:
            conn.recv()


class VecActionSpace(object):
    def __init__(self, size, action_space):
        self.size = size
        self.action_space = action_space
        self.shape = action_space.shape

    def sample(self):
        return np.array([self.action_space.sample() for _ in range(self.size)])

    @property
    def low(self):
        return self.action_space.low

    @property
    def high(self):
        return self.action_space.high


class VecObservationSpace(object):
    def __init__(self, observation_space):
        self.shape = observation_space.shape
