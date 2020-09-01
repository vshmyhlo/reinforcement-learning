import torch.nn as nn
from batchrenorm import BatchRenorm2d

from model.layers import Activation


class FCEncoder(nn.Module):
    def __init__(self, state_space, out_features):
        assert len(state_space.shape) == 1

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_space.shape[0], out_features),
            # nn.BatchNorm1d(out_features),
            Activation(),
            nn.Linear(out_features, out_features),
            # nn.BatchNorm1d(out_features),
            Activation())

    def forward(self, input):
        input = self.layers(input)

        return input


class ConvEncoder(nn.Module):
    def __init__(self, state_space, base_channels, out_features):
        assert len(state_space.shape) == 3

        super().__init__()

        self.layers = nn.Sequential(
            # nn.BatchNorm2d(state_space.shape[2]),
            nn.Conv2d(state_space.shape[2], base_channels * 2**2, 7, stride=2, padding=3),
            Activation(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(base_channels * 2**2, base_channels * 2**3, 3, stride=2, padding=1),
            Activation(),
            nn.Conv2d(base_channels * 2**3, base_channels * 2**4, 3, stride=2, padding=1),
            Activation(),
            nn.Conv2d(base_channels * 2**4, base_channels * 2**5, 3, stride=2, padding=1),
            Activation())
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.output = nn.Sequential(
            nn.Linear(base_channels * 2**5, out_features),
            Activation())

    def forward(self, input, h, d):
        dim = input.dim()

        if dim == 5:
            b, t, c, h, w = input.size()
            input = input.reshape(b * t, c, h, w)

        assert input.dim() == 4
        input = self.layers(input)
        input = self.pool(input)
        input = input.view(input.size(0), input.size(1))
        input = self.output(input)

        if dim == 5:
            input = input.view(b, t, input.size(1))

        return input, h


class GridWorldEncoder(nn.Module):
    def __init__(self, state_space, base_channels, out_features):
        super().__init__()

        self.embedding = nn.Embedding(9, base_channels * 2**0)
        self.conv = nn.Sequential(
            nn.Conv2d(base_channels * 2**0, base_channels * 2**1, 3, bias=False),
            BatchRenorm2d(base_channels * 2**1),
            Activation(),
            nn.Conv2d(base_channels * 2**1, base_channels * 2**2, 3, bias=False),
            BatchRenorm2d(base_channels * 2**2),
            Activation(),
            nn.Conv2d(base_channels * 2**2, base_channels * 2**3, 3, bias=False),
            BatchRenorm2d(base_channels * 2**3),
            Activation())
        self.output = nn.Sequential(
            nn.Linear(base_channels * 2**3, out_features),
            Activation())

    def forward(self, input):
        dim = input.dim()
        if dim == 4:
            b, t, h, w = input.size()
            input = input.reshape(b * t, h, w)

        input = self.embedding(input)
        input = input.permute(0, 3, 1, 2)
        input = self.conv(input)
        input = input.view(input.size(0), input.size(1))
        input = self.output(input)

        if dim == 4:
            input = input.view(b, t, input.size(1))

        return input
