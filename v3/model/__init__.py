import gym
from torch import nn as nn

from model.encoder import RNNEncoder, DenseEncoder, ConvEncoder, GridworldEncoder
from model.layers import NoOp
from v3.model.policy import PolicyCategorical, PolicyBeta
from v3.model.value_function import ValueFunction


# TODO: check errors in flattening


class Model(nn.Module):
    def __init__(self, model, state_space, action_space):
        def build_encoder():
            if model.encoder.type == 'dense':
                encoder = DenseEncoder(state_space, model.size)
            elif model.encoder.type == 'conv':
                encoder = ConvEncoder(state_space, model.encoder.size, model.size)
            elif model.encoder.type == 'gridworld':
                encoder = GridworldEncoder(state_space, model.encoder.size, model.size)
            else:
                raise AssertionError('invalid model.encoder.type {}'.format(model.encoder.type))

            return RNNEncoder(encoder, model.size, model.size)

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

    def forward(self, input, hidden, done):
        input, hidden = self.encoder(input, hidden, done)
        dist = self.policy(input)
        value = self.value_function(input)

        return dist, value, hidden
