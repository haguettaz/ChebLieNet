from typing import Union

import torch
import torch.transforms.functional as F
from PIL import Image as PILImage
from torch import Tensor, nn


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

    def forward(self, img: Union[PILImage, Tensor]):
        """
        Args:
            img (PILImage or Tensor): image to be rotated.

        Returns:
            (PILImage or Tensor): rotated image.
        """
        k = self.get_params()
        return F.rotate(img, k * 90)
