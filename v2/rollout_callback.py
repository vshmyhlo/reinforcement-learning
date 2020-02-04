from v2.observable import Observable
from history import History


class FullEpisodeRolloutCallback(Observable):
    def __init__(self, config, callbacks):
        super().__init__(callbacks)

        assert config.workers == 1

        self.history = History()

    def on_episode_step(self, s, a, r, d, s_prime, step):
        self.history.append(state=s, action=a, reward=r, done=d)

        if d:
            rollout = self.history.build_rollout(s_prime)
            self.rollout_done(rollout, step)
            self.history = History()

    def rollout_done(self, *args, **kwargs):
        self.run_callback('on_rollout_done', *args, **kwargs)


class FiniteHorizonRolloutCallback(Observable):
    def __init__(self, config, callbacks):
        super().__init__(callbacks)

        self.config = config
        self.history = History()

    def on_episode_step(self, s, a, r, d, s_prime, step):
        self.history.append(state=s, action=a, reward=r, done=d)

        if len(self.history) == self.config.rollout.horizon:
            rollout = self.history.build_rollout(s_prime)
            self.rollout_done(rollout, step)
            self.history = History()

    def rollout_done(self, *args, **kwargs):
        self.run_callback('on_rollout_done', *args, **kwargs)
