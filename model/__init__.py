import gym
from torch import nn as nn

from model.encoder import GridWorldEncoder, ConvEncoder, FCEncoder
from model.policy import PolicyCategorical, PolicyBeta, PolicyNormal
from model.rnn import RNN
from model.value_function import ValueFunction


class Model(nn.Module):
    def __init__(self, model, state_space, action_space):
        def build_encoder():
            if model.encoder.type == "fc":
                return FCEncoder(state_space, model.encoder.out_features)
            elif model.encoder.type == "conv":
                return ConvEncoder(
                    state_space, model.encoder.base_channels, model.encoder.out_features
                )
            elif model.encoder.type == "gridworld":
                return GridWorldEncoder(
                    state_space, model.encoder.base_channels, model.encoder.out_features
                )
            else:
                raise AssertionError("invalid type {}".format(model.encoder.type))

        def build_policy():
            if isinstance(action_space, gym.spaces.Discrete):
                return PolicyCategorical(model.encoder.out_features, action_space)
            elif isinstance(action_space, gym.spaces.Box):
                return PolicyBeta(model.encoder.out_features, action_space)
                # return PolicyNormal(model.encoder.out_features, action_space)
            else:
                raise AssertionError("invalid action_space {}".format(action_space))

        def build_value_function():
            return ValueFunction(model.encoder.out_features)

        super().__init__()

        self.encoder = build_encoder()
        self.rnn = RNN(model.rnn.type, model.encoder.out_features, model.encoder.out_features)
        self.policy = build_policy()
        self.value_function = build_value_function()

        self.apply(self.weight_init)

    def forward(self, input, h, d):
        input = self.encoder(input)
        input, h = self.rnn(input, h, d)
        dist = self.policy(input)
        value = self.value_function(input)

        return dist, value, h

    def zero_state(self, batch_size):
        return self.rnn.zero_state(batch_size)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# TODO: refactor
class ModelDQN(nn.Module):
    def __init__(self, model, state_space, action_space):
        def build_encoder():
            if model.encoder.type == "dense":
                return FCEncoder(state_space, model.size)
            elif model.encoder.type == "conv":
                return ConvEncoder(state_space, model.encoder.size, model.size)
            elif model.encoder.type == "gridworld":
                return GridWorldEncoder(state_space, model.size)
            else:
                raise AssertionError("invalid model.encoder.type {}".format(model.encoder.type))

        def build_action_value_functoin():
            return nn.Sequential(nn.Linear(model.size, action_space.n))

        super().__init__()

        self.encoder = build_encoder()
        self.action_value_function = build_action_value_functoin()

    def forward(self, input):
        input = self.encoder(input)
        action_value = self.action_value_function(input)

        return action_value
