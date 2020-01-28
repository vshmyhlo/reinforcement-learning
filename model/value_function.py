from torch import nn as nn


class ValueFunction(nn.Sequential):
    def __init__(self, in_features):
        super().__init__()

        self.layers = nn.Linear(in_features, 1)

    def forward(self, input):
        input = self.layers(input)
        input = input.squeeze(-1)

        return input
