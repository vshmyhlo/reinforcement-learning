# from PIL import  Image

import torchvision.transforms.functional as F


class ToImage(object):
    def __call__(self, input):
        input = F.to_tensor(input)
        input = F.normalize(input, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return input
