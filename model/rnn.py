import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, in_feautures, out_fearures):
        super().__init__()

        self.cell = nn.GRUCell(in_feautures, out_fearures)

    def forward(self, input, hidden):
        hidden = self.cell(input, hidden)

        return hidden, hidden


class LSTMCell(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.cell = nn.LSTMCell(in_features, out_features)

    def forward(self, input, hidden):
        hidden = torch.chunk(hidden, 2, 1)
        hidden = self.cell(input, hidden)
        input = hidden[0]
        hidden = torch.cat(hidden, 1)

        return input, hidden

    def zero_state(self, batch_size):
        return torch.zeros(batch_size, self.cell.hidden_size * 2)


class NoOpCell(nn.Module):
    def __init__(self, in_features, out_features):
        assert in_features == out_features

        super().__init__()

        self.hidden_size = out_features

    def forward(self, input, hidden):
        return input, hidden

    def zero_state(self, batch_size):
        return torch.zeros(batch_size, 1)


# TODO: debug loop
class RNN(nn.Module):
    def __init__(self, type, in_features, out_features):
        super().__init__()

        if type == 'gru':
            self.rnn = GRUCell(in_features, out_features)
        if type == 'lstm':
            self.rnn = LSTMCell(in_features, out_features)
        elif type == 'noop':
            self.rnn = NoOpCell(in_features, out_features)
        else:
            raise ValueError('invalid type {}'.format(type))

    def forward(self, input, hidden, done):
        squeeze = False
        if input.dim() == 2 and done.dim() == 1:
            input = input.unsqueeze(1)
            done = done.unsqueeze(1)
            squeeze = True

        input, hidden = self.rnn_loop(input, hidden, done)

        if squeeze:
            input = input.squeeze(1)

        return input, hidden

    def rnn_loop(self, input, hidden, done):
        outputs = []
        for t in range(input.size(1)):
            hidden = self.reset_state(hidden, done[:, t])
            output, hidden = self.rnn(input[:, t], hidden)
            outputs.append(output)

        outputs = torch.stack(outputs, 1)

        return outputs, hidden

    def reset_state(self, h, d):
        return torch.where(d.unsqueeze(-1), torch.zeros_like(h), h)

    def zero_state(self, batch_size):
        return self.rnn.zero_state(batch_size)
