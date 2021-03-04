import random

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms import functional as F
from torchvision.transforms import functional_pil as F_pil

from .functionals import _is_numpy, _is_pil_image, pil_to_tensor


class Random90Rotation(nn.Module):
    """
    Rotate the image by 0, 90, 180 ou 270 degrees clockwise.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    """

    @staticmethod
    def get_params():
        """
        Get parameters for ``rotate`` for a random rotation by 0, 90, 180 ou 270 degrees clockwise.

        Returns:
            (int): number of 90 degrees clockwise rotations to perform.
        """
        return random.randint(0, 3)

    def forward(self, input):
        """
        Args:
            input (PILImage or Tensor): input to rotate.

        Returns:
            (PILImage or Tensor): rotated output.
        """
        k = self.get_params()
        return F.rotate(input, k * 90)


class Normalize:
    """
    Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    """

    def __init__(self, mean, std):
        """
        Args:
            mean (sequence): sequence of means for each channel.
            std (sequence): sequence of standard deviations for each channel.
        """
        self.mean = Tensor(mean).unsqueeze(1)
        self.std = Tensor(std).unsqueeze(1)

    def forward(self, input):
        """
        Args:
            input (`torch.Tensor`): input tensor with size (C, V) to be normalized.

        Returns:
            (`torch.Tensor`): normalized output tensor.
        """
        return (input - self.mean) / self.std


class ToTensor:
    """
    Convert input object to PyTorch tensor.
    """

    def forward(self, input):
        """
        Args:
            input (:obj:): input object to convert.

        Returns:
            (`torch.Tensor`): output tensor with size (1, C, V).
        """

        if _is_pil_image(input):
            return pil_to_tensor(input)

        if _is_numpy(input):
            return torch.from_numpy(input)

        return Tensor(input)


class Compose:
    """
    Compose several transforms together. This transform does not support torchscript.
    """

    def __init__(self, transforms):
        """
        Args:
            transforms (list): list of transformations to compose.
        """

        self.transforms = transforms

    def forward(self, input):
        """
        Args:
            input (:obj:): input object to transform.

        Returns:
            (:obj:): transformed output object.
        """
        for t in self.transforms:
            input = t(input)
        return input

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToGEGraphSignal:
    """
    Make a tensor a group equivariant graph signal with shape (C, V) where C corresponds to the number of channels
    and V the number of vertices. The initial signal is repeated along the symmetric layers, such that the final number
    of vertices is V = L * Vin.
    """

    def __init__(self, num_layers):
        """
        Args:
            num_layers (int): number of symmetric layers.
        """
        self.num_layers = num_layers

    def forward(self, input):
        """
        Args:
            input (`torch.Tensor`): input tensor to convert.

        Returns:
            (`torch.Tensor`): output tensor with shape (C, V).
        """

        if input.ndim < 2:
            input = input.unsqueeze(0)

        C, *_ = input.shape

        return input.reshape(C, -1).unsqueeze(1).expand(-1, self.num_layers, -1).reshape(C, -1)


class BoolToInt:
    """
    Convert BoolTensor with shape (B, C, ...) to IntTensor with shape (B, 1, ...).
    """

    def forward(self, input):
        """
        Args:
            input (`torch.Tensor`): input tensor to convert.
        Returns:
            (`torch.Tensor`): output tensor.
        """

        return input.int().argmax(dim=0)
