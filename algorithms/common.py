import torch
import torchvision.transforms as T

from transforms import ToImage


def build_optimizer(optimizer, parameters):
    if optimizer.type == 'momentum':
        return torch.optim.SGD(parameters, optimizer.lr, momentum=0.9, weight_decay=0.)
    elif optimizer.type == 'rmsprop':
        return torch.optim.RMSprop(parameters, optimizer.lr, weight_decay=0.)
    elif optimizer.type == 'adam':
        return torch.optim.Adam(parameters, optimizer.lr, weight_decay=0.)
    else:
        raise AssertionError('invalid optimizer.type {}'.format(optimizer.type))


def build_transform(transform):
    if transform == 'noop':
        transform = T.Compose([])
    elif transform == 'image':
        transform = ToImage()
    else:
        raise AssertionError('invalid transform {}'.format(transform))

    return transform
