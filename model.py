import torch
import torch.nn as nn


class NoOp(nn.Module):
    def forward(self, input):
        return input


class Activation(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels))


class Model(nn.Module):
    def __init__(self, model, state_shape, num_actions):
        def build_encoder():
            if model.encoder.type == 'dense':
                assert len(state_shape) == 1
                return Encoder(state_shape[0], model.size)
            elif model.encoder.type == 'conv':
                assert len(state_shape) == 3
                return ConvEncoder(state_shape[2], model.encoder.size, model.size)
            else:
                raise AssertionError('invalid model.encoder.type {}'.format(model.encoder.type))

        super().__init__()

        if model.encoder.shared:
            self.encoder = build_encoder()
            self.policy = PolicyCategorical(model.size, num_actions)
            self.value_function = ValueFunction(model.size)
        else:
            self.encoder = NoOp()
            self.policy = nn.Sequential(
                build_encoder(),
                PolicyCategorical(model.size, num_actions))
            self.value_function = nn.Sequential(
                build_encoder(),
                ValueFunction(model.size))

    def forward(self, input):
        input = self.encoder(input)
        dist = self.policy(input)
        value = self.value_function(input)

        return dist, value


class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            Activation(),
            nn.Linear(out_features, out_features),
            Activation())

    def forward(self, input):
        input = self.layers(input)

        return input


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, out_features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            ConvNorm(in_channels, base_channels * 2**2, 7, stride=2, padding=7 // 2),
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


class ValueFunction(nn.Sequential):
    def __init__(self, in_features):
        super().__init__()

        self.layers = nn.Linear(in_features, 1)

    def forward(self, input):
        input = self.layers(input)
        input = input.squeeze(-1)

        return input


class PolicyCategorical(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()

        self.layers = nn.Linear(in_features, num_actions)

    def forward(self, input):
        input = self.layers(input)
        dist = torch.distributions.Categorical(logits=input)

        return dist
