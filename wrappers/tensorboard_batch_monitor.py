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

        # FIXME: last render includes scene from next episode
       
        for i in self.tracks:
            frame = self.env.render(mode='rgb_array', index=i).copy()
            frame = torch.tensor(frame).permute(2, 0, 1)
            self.tracks[i].frames.append(frame)

        indices, = np.where(done)
        for i in indices:
            self.episodes += 1

            if i in self.tracks:
                print('finished: episode_number {}, index_in_batch {}'
                      .format(self.tracks[i].episode_number, i))

                fps = self.env.metadata.get('video.frames_per_second') or 24
                fps = min(fps, 60)

                self.writer.add_video(
                    'episode',
                    torch.stack(self.tracks[i].frames, 0).unsqueeze(0),
                    fps=fps,
                    global_step=self.tracks[i].episode_number)
                self.writer.flush()

                del self.tracks[i]

            if self.episodes % self.log_interval == 0:
                assert i not in self.tracks
                self.tracks[i] = self.Track(
                    episode_number=self.episodes,
                    frames=[])

                print('tracking: episode_number {}, index_in_batch {}'
                      .format(self.tracks[i].episode_number, i))

        return state, reward, done, meta
