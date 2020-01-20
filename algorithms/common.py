import torch


def build_optimizer(optimizer, parameters, learning_rate):
    if optimizer == 'momentum':
        return torch.optim.SGD(parameters, learning_rate, momentum=0.9, weight_decay=0.)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters, learning_rate, weight_decay=0.)
    elif optimizer == 'adam':
        return torch.optim.Adam(parameters, learning_rate, weight_decay=0.)
    else:
        raise AssertionError('invalid optimizer {}'.format(optimizer))
