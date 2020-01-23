import torch
import torch.nn as nn

SIZE = 32


class Activation(nn.PReLU):
    pass


class Model(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()

        self.policy = nn.Sequential(
            Embedder(state_size),
            PolicyCategorical(num_actions))

        self.value_function = nn.Sequential(
            Embedder(state_size),
            ValueFunction())


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

        self.linear = nn.Linear(in_features, SIZE)
        self.act = Activation()

    def forward(self, input):
        input = self.linear(input)
        input = self.act(input)

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
