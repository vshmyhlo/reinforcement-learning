# Implementations of popular deep reinforcement learning algorithms

* [REINFORCE (Policy Gradient Monte Carlo)](v1/pg_mc.py)
* [Actor Critic with Monte Carlo advantage estimate](v1/ac_mc.py)
* [Advantage Actor Critic (A2C)](v1/a2c.py)

# TODO
* ideas from atari preprocessing
* normalize input
* skip-observations
* optimize for speed
* rnn-model
* 4-frames stack
* plot grad dist/grad norm
* plot different losses
* plot more metrics (from shultz presentation)
* plot mean/std
* mean by time
* remove float casts
* refactor rollout to use s_prime at every step
* use record episode stats
* merge wrappers and transforms
* make layers shared between versions
* check all conv paddings
* 5 step horizon
* use activation for value prediction
* add action to obs
* advantage normalization
* td(0)
* exp replay
* gae
* td(lambda)
* mpi
* a3c
* compute running mean/std of metrics
