from collections import namedtuple

import gym
import numpy as np
import torch


class Torch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)

        self.device = device

    def reset(self):
        state = self.env.reset()

        state = torch.tensor(state, device=self.device)

        return state

    def step(self, action):
        action = action.data.cpu().numpy()

        state, reward, done, meta = self.env.step(action)

        state = torch.tensor(state, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)

        return state, reward, done, meta


class Batch(gym.Wrapper):
    def reset(self, **kwargs):
        state = self.env.reset()

        state = np.expand_dims(np.array(state), 0)

        return state

    def step(self, action):
        action = np.squeeze(action, 0)

        state, reward, done, meta = self.env.step(action)

        state = np.expand_dims(np.array(state), 0)
        reward = np.expand_dims(np.array(reward), 0)
        done = np.expand_dims(np.array(done), 0)
        meta = [meta]

        return state, reward, done, meta

    def render(self, mode='human', index=0):
        assert index == 0

        return self.env.render(mode=mode)


class StackObservation(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)

        self.k = k
        self.buffer = None
        self.observation_space = gym.spaces.Box(
            low=np.expand_dims(self.observation_space.low, -1).repeat(self.k, -1),
            high=np.expand_dims(self.observation_space.high, -1).repeat(self.k, -1),
            dtype=self.observation_space.dtype)

    def reset(self, **kwargs):
        self.buffer = []

        state = self.env.reset()
        self.buffer.append(state)

        while len(self.buffer) < self.k:
            state, reward, done, meta = self.env.step(self.action_space.sample())
            self.buffer.append(state)

        state = np.stack(self.buffer, -1)

        return state

    def step(self, action):
        action = np.squeeze(action, 0)

        state, reward, done, meta = self.env.step(action)
        self.buffer.append(state)
        self.buffer = self.buffer[-self.k:]

        state = np.stack(self.buffer, -1)

        return state, reward, done, meta


class TensorboardBatchMonitor(gym.Wrapper):
    Track = namedtuple('Track', ['number', 'index', 'frames'])

    def __init__(self, env, writer, log_interval):
        super().__init__(env)

        self.writer = writer
        self.log_interval = log_interval

        self.episodes = 0
        self.track = None

    def step(self, action):
        state, reward, done, meta = self.env.step(action)

        if self.track is not None:
            frame = self.env.render(mode='rgb_array', index=self.track.index).copy()
            frame = torch.tensor(frame).permute(2, 0, 1)
            self.track.frames.append(frame)

        indices, = np.where(done)
        for i in indices:
            self.episodes += 1

            if self.track is None:
                if self.episodes % self.log_interval == 0:
                    self.track = self.Track(
                        number=self.episodes,
                        index=i,
                        frames=[])

                    print('tracking: episode {}, index {}'.format(self.track.number, self.track.index))
            else:
                if i == self.track.index:
                    print('finished: episode {}, index {}'.format(self.track.number, self.track.index))

                    fps = self.env.metadata.get('video.frames_per_second') or 24
                    fps = min(fps, 60)

                    self.writer.add_video(
                        'episode',
                        torch.stack(self.track.frames, 0).unsqueeze(0),
                        fps=fps,
                        global_step=self.track.number)
                    self.writer.flush()

                    self.track = None

        return state, reward, done, meta
