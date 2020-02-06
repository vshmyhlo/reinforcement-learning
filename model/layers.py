from torch import nn as nn


class NoOp(nn.Module):
    def forward(self, input):
        return input


class Activation(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)


class ConvNorm(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels))
