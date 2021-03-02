import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms import functional as F
from torchvision.transforms import functional_pil as F_pil

from .functionals import _is_numpy, _is_pil_image, pil_to_tensor


class Random90Rotation(nn.Module):
    """
    Rotate the image by k-90 angle.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params() -> int:
        """Get parameters for ``rotate`` for a random k-90 degrees rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random k-90 rotation.
        """
        k = int(torch.empty(1).random_(0, 4).item())
        return k

    def forward(self, item):
        """
        Args:
            item (PILImage or Tensor): image to be rotated.

        Returns:
            (PILImage or Tensor): rotated image.
        """
        k = self.get_params()
        return F.rotate(item, k * 90)


class Normalize:
    """Normalize using mean and std."""

    def __init__(self, mean, std):
        """
        Initialization.

        Args:
            mean (tuple): means of each feature.
            std (tuple): standard deviations of each feature.
        """
        self.mean = torch.tensor(mean).unsqueeze(1)
        self.std = torch.tensor(std).unsqueeze(1)

    def __call__(self, item):
        """
        Calling function.

        Args:
            item (`torch.Tensor`): sample of size (1, features, vertices) to be normalized on its features.

        Returns:
            `torch.Tensor`: normalized input tensor.
        """
        return (item - self.mean) / self.std


class ToTensor:
    """Convert raw data and labels to PyTorch tensor."""

    def __call__(self, item):
        """Function call operator to change type.
        Args:
            item (`numpy.array`): input that needs to be transformed.
        Returns:
            `torch.Tensor`: sample of size (1, features, vertices).
        """

        if _is_pil_image(item):
            return pil_to_tensor(item)

        if _is_numpy(item):
            return torch.from_numpy(item)

        return torch.Tensor(item)


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToGraphSignal:
    """Convert raw data and labels to PyTorch tensor."""

    def __init__(self, num_layers):
        """
        Initialization.

        Args:
            num_layers (int): .
        """
        self.num_layers = num_layers

    def __call__(self, item):
        """Function call operator to change type.
        Args:
            item (`torch.Tensor`): input that needs to be transformed.
        Returns:
            `torch.Tensor`: sample of size (1, features, vertices).
        """

        if item.ndim < 2:
            item = item.unsqueeze(0)

        C, *_ = item.shape

        return item.reshape(C, -1).unsqueeze(1).expand(-1, self.num_layers, -1).reshape(C, -1)


class BoolToInt:
    """Convert raw data and labels to PyTorch tensor."""

    def __call__(self, item):
        """Function call operator to change type.
        Args:
            item (`torch.Tensor`): input that needs to be transformed.
        Returns:
            `torch.Tensor`: sample of size (1, features, vertices).
        """

        return item.int().argmax(dim=0)
