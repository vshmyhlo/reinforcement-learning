from config import CN


def build_default_config():
    config = CN()

    config.seed = 42
    config.env = 'CartPole-v0'
    config.steps = 1000000
    config.log_interval = 10000
    config.transforms = []
    config.gamma = 0.99
    config.entropy_weight = 1e-2
    config.workers = 1

    config.rollout = CN()
    config.rollout.type = 'full_episode'

    config.model = CN()
    config.model.size = 32

    config.model.encoder = CN()
    config.model.encoder.type = 'dense'
    config.model.encoder.size = 32
    config.model.encoder.shared = True

    config.opt = CN()
    config.opt.type = 'adam'
    config.opt.lr = 1e-3

    return config
