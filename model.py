import torch
import torch.nn as nn

SIZE = 32


class Model(nn.Module):
    def __init__(self, policy=None, value_function=None):
        super().__init__()

        if policy is not None:
            self.policy = policy
        if value_function is not None:
            self.value_function = value_function


class Encoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.l_1 = nn.Linear(in_features, SIZE)
        self.l_2 = nn.Linear(SIZE, SIZE)
        self.act = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.l_1.weight)
        nn.init.xavier_normal_(self.l_2.weight)

    def forward(self, input):
        input = self.l_1(input)
        input = self.act(input)
        input = self.l_2(input)
        input = self.act(input)

        return input


class ValueFunction(nn.Module):
    def __init__(self, state_size):
        super().__init__()

        self.net = Encoder(state_size)
        self.linear = nn.Linear(SIZE, 1)

        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, input):
        input = self.net(input)
        input = self.linear(input)
        input = input.squeeze(-1)

        return input


class PolicyCategorical(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()

        self.net = Encoder(state_size)
        self.dense = nn.Linear(SIZE, num_actions)

        nn.init.xavier_normal_(self.dense.weight)

    def forward(self, input):
        input = self.net(input)
        input = self.dense(input)
        dist = torch.distributions.Categorical(logits=input)

        return dist
