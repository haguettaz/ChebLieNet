import math
from typing import Tuple, Union

from torch import Tensor, nn


class CubicPool(nn.Module):
    """
    Basic class for general pooling on a cubic graph - well-adapted for SE(2) group.
    """

    def __init__(self, kernel_size: Tuple[int, int], size: Tuple[int, int, int]):
        """
        Initialization.

        Args:
            kernel_size (tuple of int): pooling reduction in format (spatial_red, sym_red).
        """
        super(CubicPool, self).__init__()

        self.size = size  # format (L, H, W)
        self.shortcut = kernel_size == (1, 1)
        self.maxpool = nn.MaxPool3d((kernel_size[0], kernel_size[1], kernel_size[1]))

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

        B, C, _ = x.shape

        x = x.reshape(B, C, *self.size)
        x = self.maxpool(x)
        return x.reshape(B, C, -1)


class IcosahedralPool(nn.Module):
    """
    Basic class for general pooling on a spherical graph - well-adapted for SO(3) group.
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
        super(IcosahedralPool, self).__init__()

        self.size = size  # (L, V)
        lvl_in = int(math.log((size[1] - 2) / 10) / math.log(4))
        lvl_out = lvl_in - int(math.log(kernel_size[1]) / math.log(2))
        self.dim_out = (size[0] // kernel_size[0], 1 if lvl_out < 0 else int(10 * 4 ** lvl_out + 2))

        self.sym_shortcut = kernel_size[0] == 1
        self.spatial_shortcut = kernel_size[1] == 1

        self.maxpool = nn.MaxPool2d((kernel_size[0], 1))

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

        B, C, _ = x.shape

        x = x.reshape(B, C, *self.size)

        if not self.spatial_shortcut:
            x = x[..., : self.dim_out[1]]

        if not self.sym_shortcut:
            x = self.maxpool(x)

        return x.reshape(B, C, -1)
