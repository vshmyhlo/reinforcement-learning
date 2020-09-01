from torch import nn as nn


class NoOp(nn.Module):
    def forward(self, input):
        return input


class Activation(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)
