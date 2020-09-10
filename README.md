# Implementations of popular deep reinforcement learning algorithms

* [REINFORCE (Policy Gradient Monte Carlo)](algo/pg_mc.py)
* [Actor Critic with Monte Carlo advantage estimate](algo/ac_mc.py)
* [Advantage Actor Critic (A2C)](algo/a2c.py)

# TODO
* batch-norm not working in eval mode
* ideas from atari preprocessing
* normalize input
* optimize for speed
* 4-frames stack
* plot grad dist/grad norm
* plot different losses
* plot more metrics (from shultz presentation)
* mean by time
* remove float casts
* refactor rollout to use s_prime at every step
* normalize input
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
* td(lambda)
* mpi
* a3c
* compute running mean/std of metrics
* rename meta to info
