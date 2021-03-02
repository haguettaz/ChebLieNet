import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CubicUnpool(nn.Module):
    """
    Basic class for general unpooling on a cubic graph - well-adapted for SE(2) group.
    """

    def __init__(self, kernel_size: Tuple[int, int], size: Tuple[int, int, int]):
        """
        Initialization.

        Args:
            kernel_size (tuple of int): pooling reduction in format (spatial_red, sym_red).
        """
        super(CubicUnpool, self).__init__()

        self.size = size  # format (L, H, W)
        self.dim_out = (size[0] * kernel_size[0], size[1] * kernel_size[1], size[1] * kernel_size[1])
        self.shortcut = kernel_size == (1, 1)

        self.avgunpool = nn.AdaptiveAvgPool3d(self.dim_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """
        if self.shortcut:
            return x

        B, C, *_ = x.shape

        x = x.reshape(B, C, *self.size)

        x = self.avgunpool(x)

        return x.reshape(B, C, -1)


class IcosahedralUnpool(nn.Module):
    """
    Basic class for general unpooling on a spherical graph - well-adapted for SO(3) group.
    Pooling is based on the assumption the vertices are sorted the same way as Max Jiang.
    See: https://github.com/maxjiang93/ugscnn/blob/master/meshcnn/mesh.py
    """

    def __init__(self, kernel_size: Tuple[int, int], size: Tuple[int, int]):
        """
        Initialization.

        Args:
            Lin (int): input depth.
            Lout (int): ouput depth.
            lvlin (int): input icosahedron level.
            lvlout (int): output icosahedron level.
        """
        super(IcosahedralUnpool, self).__init__()

        self.size = size  # (L, V)
        lvl_in = int(math.log((size[1] - 2) / 10) / math.log(4))
        lvl_out = lvl_in + int(math.log(kernel_size[1]) / math.log(2))
        self.dim_out = (size[0] * kernel_size[0], int(10 * 4 ** lvl_out + 2))

        self.sym_shortcut = kernel_size[0] == 1
        self.spatial_shortcut = kernel_size[1] == 1

        self.maxunpool = nn.AdaptiveAvgPool2d(self.dim_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """

        if self.spatial_shortcut and self.sym_shortcut:
            return x

        B, C, *_ = x.shape

        x = x.reshape(B, C, *self.size)

        if not self.spatial_shortcut:
            x = F.pad(x, (0, self.dim_out[1] - self.size[1]), "constant", value=0.0)

        if not self.sym_shortcut:
            x = self.maxunpool(x)

        return x.reshape(B, C, -1)
