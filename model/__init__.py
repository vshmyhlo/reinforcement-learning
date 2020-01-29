import gym
from torch import nn as nn

from model.layers import Activation, ConvNorm, NoOp
from model.policy import PolicyCategorical, PolicyBeta
from model.value_function import ValueFunction


class Model(nn.Module):
    def __init__(self, model, state_space, action_space):
        def build_encoder():
            if model.encoder.type == 'dense':
                return DenseEncoder(state_space, model.size)
            elif model.encoder.type == 'conv':
                return ConvEncoder(state_space, model.encoder.size, model.size)
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


class DenseEncoder(nn.Module):
    def __init__(self, state_space, out_features):
        assert len(state_space.shape) == 1

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_space.shape[0], out_features),
            Activation(),
            nn.Linear(out_features, out_features),
            Activation())

    def forward(self, input):
        input = self.layers(input)

        return input


class ConvEncoder(nn.Module):
    def __init__(self, state_space, base_channels, out_features):
        assert len(state_space.shape) == 3

        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(state_space.shape[2]),
            ConvNorm(state_space.shape[2], base_channels * 2**2, 7, stride=2, padding=7 // 2),
            Activation(),
            nn.MaxPool2d(3, 2),
            ConvNorm(base_channels * 2**2, base_channels * 2**3, 3, stride=2, padding=3 // 2),
            Activation(),
            ConvNorm(base_channels * 2**3, base_channels * 2**4, 3, stride=2, padding=3 // 2),
            Activation(),
            ConvNorm(base_channels * 2**4, base_channels * 2**5, 3, stride=2, padding=3 // 2),
            Activation())
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.output = nn.Linear(base_channels * 2**5, out_features)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        dim = input.dim()
        if dim == 3:
            input = input.unsqueeze(0)
        elif dim == 5:
            b, t, c, h, w = input.size()
            input = input.reshape(b * t, c, h, w)

        assert input.dim() == 4
        input = self.layers(input)
        input = self.pool(input)
        input = input.view(input.size(0), input.size(1))
        input = self.output(input)

        if dim == 3:
            input = input.squeeze(0)
        elif dim == 5:
            input = input.view(b, t, input.size(1))

        return input
