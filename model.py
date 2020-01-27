import torch
import torch.nn as nn

SIZE = 32


class Activation(nn.PReLU):
    pass


# TODO: atari shared

class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels))


class ConvEmbedder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        base = 16

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            ConvNorm(in_channels, base * 2**0, 7, stride=2, padding=7 // 2),
            Activation(),
            nn.MaxPool2d(3, 2),
            ConvNorm(base * 2**0, base * 2**1, 3, stride=2, padding=3 // 2),
            Activation(),
            ConvNorm(base * 2**1, base * 2**2, 3, stride=2, padding=3 // 2),
            Activation(),
            ConvNorm(base * 2**2, base * 2**3, 3, stride=2, padding=3 // 2),
            Activation())
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.output = nn.Linear(base * 2**3, SIZE)

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


class Model(nn.Module):
    def __init__(self, state_shape, num_actions):
        def build_embedder():
            if len(state_shape) == 1:
                return Embedder(state_shape[0])
            else:
                return ConvEmbedder(state_shape[2])

        super().__init__()

        self.policy = nn.Sequential(
            build_embedder(),
            PolicyCategorical(num_actions))

        self.value_function = nn.Sequential(
            build_embedder(),
            ValueFunction())

    def forward(self, input):
        dist = self.policy(input)
        value = self.value_function(input)

        return dist, value


class ModelRNN(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()

        self.policy_embedder = EmbedderRNN(state_size)
        self.policy_module = PolicyCategorical(num_actions)

        self.value_function_embedder = EmbedderRNN(state_size)
        self.value_function_module = ValueFunction()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def policy(self, input, hidden):
        input, hidden = self.policy_embedder(input, hidden)
        input = self.policy_module(input)

        return input, hidden

    def value_function(self, input, hidden):
        input, hidden = self.value_function_embedder(input, hidden)
        input = self.value_function_module(input)

        return input, hidden


class EmbedderRNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.rnn = nn.LSTM(in_features, SIZE, batch_first=True, bidirectional=False)

    def forward(self, input, hidden):
        dim = input.dim()

        if dim == 1:
            input = input.view(1, 1, input.size(0))
        elif dim == 2:
            input = input.unsqueeze(1)

        assert input.dim() == 3
        input, hidden = self.rnn(input, hidden)

        if dim == 1:
            input = input.view(input.size(2))
        elif dim == 2:
            input = input.squeeze(1)

        return input, hidden


class Embedder(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, SIZE),
            Activation(),
            nn.Linear(SIZE, SIZE),
            Activation())

    def forward(self, input):
        input = self.layers(input)

        return input


class ValueFunction(nn.Sequential):
    def __init__(self):
        super().__init__()

        self.layers = nn.Linear(SIZE, 1)

    def forward(self, input):
        input = self.layers(input)
        input = input.squeeze(-1)

        return input


class PolicyCategorical(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.layers = nn.Linear(SIZE, num_actions)

    def forward(self, input):
        input = self.layers(input)
        dist = torch.distributions.Categorical(logits=input)

        return dist
