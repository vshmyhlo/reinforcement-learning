import gym


# TODO: what to do with render?
class SkipObservation(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)

        self.k = k

    def step(self, action):
        reward_buffer = 0
        for _ in range(self.k):
            state, reward, done, meta = self.env.step(action)
            reward_buffer += reward

            if done:
                break

        return state, reward_buffer, done, meta
