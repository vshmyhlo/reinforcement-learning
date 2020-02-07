from ruamel.yaml import YAML


class CN(dict):
    IMMUTABLE = '__immutable__'

    def __init__(self):
        super().__init__()

        self[self.IMMUTABLE] = False

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        assert not self[self.IMMUTABLE]

        self[key] = value

    def freeze(self):
        self[self.IMMUTABLE] = True

        for k in self:
            if isinstance(self[k], CN):
                self[k].freeze()

    def merge_from_config(self, config):
        for k in config:
            assert k in self
           
            if isinstance(config[k], CN):
                self[k].merge_from_config(config[k])
            else:
                self[k] = config[k]

    def merge_from_file(self, path):
        with open(path) as f:
            config = YAML().load(f)

        config = self.coerce(config)
        assert isinstance(config, CN)
        self.merge_from_config(config)

    @classmethod
    def coerce(cls, value):
        if isinstance(value, dict):
            return cls.build_from_dict(value)
        elif isinstance(value, list):
            return [cls.coerce(item) for item in value]
        else:
            return value

    @classmethod
    def build_from_dict(cls, data):
        config = CN()

        for k in data:
            config[k] = cls.coerce(data[k])

        return config
