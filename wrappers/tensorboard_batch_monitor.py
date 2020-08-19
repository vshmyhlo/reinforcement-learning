from collections import namedtuple

import gym
import numpy as np
import torch


class TensorboardBatchMonitor(gym.Wrapper):
    Track = namedtuple('Track', ['episode_number', 'frames'])

    def __init__(self, env, writer, log_interval):
        super().__init__(env)

        self.writer = writer
        self.log_interval = log_interval

        self.episodes = 0
        self.tracks = {}

    def step(self, action):
        state, reward, done, meta = self.env.step(action)

        for i in self.tracks:
            frame = self.env.render(mode='rgb_array', index=i).copy()
            frame = torch.tensor(frame).permute(2, 0, 1)
            self.track.frames.append(frame)

        indices, = np.where(done)
        for i in indices:
            self.episodes += 1

            if self.track is None:
                if self.episodes % self.log_interval == 0:
                    self.tracks[i] = self.Track(
                        episode_number=self.episodes,
                        frames=[])

                    print('tracking: episode_number {}, index_in_batch {}'
                          .format(self.track.episode_number, i))
            else:
                if i in self.tracks:
                    print('finished: episode_number {}, index_in_batch {}'
                          .format(self.track.episode_number, i))

                    fps = self.env.metadata.get('video.frames_per_second') or 24
                    fps = min(fps, 60)

                    self.writer.add_video(
                        'episode',
                        torch.stack(self.track.frames, 0).unsqueeze(0),
                        fps=fps,
                        global_step=self.track.episode_number)
                    self.writer.flush()

                    self.track = None

        return state, reward, done, meta
