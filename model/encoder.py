import torch
import torch.nn as nn

from model.layers import ConvNorm, Activation


class FCEncoder(nn.Module):
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

    def zero_state(self, batch_size):
        return torch.zeros(batch_size, 1)


class ConvRNNEncoder(nn.Module):
    def __init__(self, state_space, base_channels, out_features):
        super().__init__()

        self.conv = ConvEncoder(state_space, base_channels, out_features)
        self.rnn = nn.LSTMCell(out_features, out_features)

    def forward(self, input, h, d):
        h = self.reset_state(h, d)

        input, _ = self.conv(input, None, None)
        h = self.rnn(input, h)
        input, _ = h

        return input, h

    def reset_state(self, h, d):
        if h is None or d is None:
            assert h is None
            assert d is None
        else:
            h = tuple(torch.where(d.unsqueeze(-1), torch.zeros_like(x), x) for x in h)

        return h


class GridworldEncoder(nn.Module):
    def __init__(self, state_space, base_channels, out_features):
        super().__init__()

        self.embedding = nn.Embedding(10, base_channels * 2**0)
        self.conv = nn.Sequential(
            ConvNorm(base_channels * 2**0, base_channels * 2**1, 3, 2),
            Activation(),
            ConvNorm(base_channels * 2**1, base_channels * 2**2, 3),
            Activation())
        self.output = nn.Linear(base_channels * 2**2, out_features)

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


class CustomRNNCell(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(features * 2, features),
            nn.Tanh())
        self.w = nn.Sequential(
            nn.Linear(features * 2, features),
            nn.Sigmoid())

        nn.init.constant_(self.w[0].bias, 10.)

    def forward(self, input, hidden):
        if hidden is None:
            hidden = torch.zeros_like(input)

        cat = torch.cat([input, hidden], -1)
        hidden = self.hidden(cat)
        w = self.w(cat)

        return w * input + (1 - w) * hidden


class RNNEncoder(nn.Module):
    def __init__(self, encoder, in_features, out_features):
        super().__init__()

        self.encoder = encoder
        # self.rnn = nn.GRUCell(in_features, out_features)
        self.rnn = CustomRNNCell(in_features)
        self.output = nn.Sequential()

    def forward(self, input, hidden, done):
        input = self.encoder(input)

        dim = input.dim()
        if dim == 2:
            input = input.unsqueeze(1)
            assert done is None
        else:
            done = done.unsqueeze(-1)

        outputs = []
        for t in range(input.size(1)):
            hidden = self.rnn(input[:, t], hidden)
            outputs.append(hidden)
            if done is not None:
                hidden = torch.where(done[:, t], torch.zeros_like(hidden), hidden)

        input = torch.stack(outputs, 1)
        input = self.output(input)

        if dim == 2:
            input = input.squeeze(1)

        return input, hidden
