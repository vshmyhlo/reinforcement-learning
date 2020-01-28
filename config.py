from yacs.config import CfgNode as CN


def build_default_config():
    config = CN()

    config.seed = 42
    config.env = 'CartPole-v0'
    config.episodes = 1000
    config.log_interval = 100
    config.transform = 'noop'
    config.gamma = 0.99
    config.entropy_weight = 1e-2
    config.horizon = 8
    config.workers = 32

    config.model = CN()
    config.model.size = 32

    config.model.encoder = CN()
    config.model.encoder.type = 'dense'
    config.model.encoder.size = 16
    config.model.encoder.shared = True

    config.opt = CN()
    config.opt.type = 'adam'
    config.opt.lr = 1e-3

    return config
