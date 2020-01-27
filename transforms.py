import numba
import numpy as np


class ToImage(object):
    def __call__(self, input):
        input = np.moveaxis(input, 2, 0)
        input = self.normalize(input)

        return input

    @staticmethod
    @numba.njit()
    def normalize(input):
        input = input.astype(np.float32)
        input -= 255 / 2
        input /= 255 / 2

        return input
