from ruamel.yaml import YAML


class CN(dict):
    def __init__(self):
        super().__init__()

        self.frozen = False

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def freeze(self):
        assert not self.frozen
        self.frozen = True

    def merge_from_file(self, path):
        return

        with open(path) as f:
            config = self.build_from_dict(YAML().load(f))
            pass

    def check_types(self, a, b):
        pass

    @classmethod
    def build_from_dict(cls, data):
        return
        config = CN()

        for k in data:
            if isinstance(data[k], dict):
                config[k] = cls.build_from_dict(data[k])
            else:
                config[k] = data[k]


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
