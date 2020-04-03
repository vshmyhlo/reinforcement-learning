import gym
from torch import nn as nn

from model.encoder import GridworldEncoder, ConvEncoder, DenseEncoder
from model.layers import NoOp
from v1.model.policy import PolicyCategorical, PolicyBeta
from v1.model.value_function import ValueFunction


class Model(nn.Module):
    def __init__(self, model, state_space, action_space):
        def build_encoder():
            if model.encoder.type == 'dense':
                return DenseEncoder(state_space, model.size)
            elif model.encoder.type == 'conv':
                return ConvEncoder(state_space, model.encoder.size, model.size)
            elif model.encoder.type == 'gridworld':
                return GridworldEncoder(state_space, model.size)
            else:
                raise AssertionError('invalid model.encoder.type {}'.format(model.encoder.type))

        def build_policy():
            if isinstance(action_space, gym.spaces.Discrete):
                return PolicyCategorical(model.size, action_space)
            elif isinstance(action_space, gym.spaces.Box):
                return PolicyBeta(model.size, action_space)
            else:
                raise AssertionError('invalid action_space {}'.format(action_space))

        def build_value_function():
            return ValueFunction(model.size)

        super().__init__()

        if model.encoder.shared:
            self.encoder = build_encoder()
            self.policy = build_policy()
            self.value_function = build_value_function()
        else:
            self.encoder = NoOp()
            self.policy = nn.Sequential(
                build_encoder(),
                build_policy())
            self.value_function = nn.Sequential(
                build_encoder(),
                build_value_function())

    def forward(self, input):
        input = self.encoder(input)
        dist = self.policy(input)
        value = self.value_function(input)

        return dist, value


class ModelTMP(nn.Module):
    def __init__(self, model, state_space, action_space):
        def build_encoder():
            if model.encoder.type == 'dense':
                return DenseEncoder(state_space, model.size)
            elif model.encoder.type == 'conv':
                return ConvEncoder(state_space, model.encoder.size, model.size)
            elif model.encoder.type == 'gridworld':
                return GridworldEncoder(state_space, model.size)
            else:
                raise AssertionError('invalid model.encoder.type {}'.format(model.encoder.type))

        def build_action_value_functoin():
            return nn.Sequential(
                nn.Linear(model.size, action_space.n))

        super().__init__()

        self.encoder = build_encoder()
        self.action_value_function = build_action_value_functoin()

    def forward(self, input):
        input = self.encoder(input)
        action_value = self.action_value_function(input)

        return action_value
