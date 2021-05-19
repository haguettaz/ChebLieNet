# coding=utf-8

import io
import math
import os
import pkgutil

import torch
from torch import nn

from ...utils.utils import mod


def avg_unpool(x, index):
    out = x[..., index].mean(dim=-1)
    return out


def max_unpool(x, index):
    out, _ = x[..., index].max(dim=-1)
    return out


def rand_unpool(x, index):
    N, K = index.shape
    out = x[..., index[torch.arange(N), torch.randint(K, (N,))]]
    return out


class SE2SpatialUnpool(nn.Module):
    """
    A SE(2) spatial unpooling layer. Required the  to be ordered L, H, W.
    """

    def __init__(self, kernel_size, size, expansion):
        """
        Args:
            kernel_size (int): pooling reduction size.
            size (sequence of ints): size in format (nx, ny, ntheta)
            expansion (str): expansion operation.
        """
        super(SE2SpatialUnpool, self).__init__()

        if expansion not in {"max", "avg", "rand"}:
            raise ValueError(f"{expansion} is not a valid value for expansion, must be 'max' 'avg' or 'rand'.")

        self.index = self.get_expansion_index(size, kernel_size)

        if expansion == "rand":
            self.expansion = rand_unpool
        elif expansion == "max":
            self.expansion = max_unpool
        else:
            self.expansion = avg_unpool

        self.shortcut = kernel_size == 1

    def get_expansion_index(self, size, kernel_size):

        nx, ny, ntheta = size

        Fout = nx * kernel_size * ny * kernel_size
        Vout = Fout * ntheta

        indices = torch.arange(Vout)

        ix = mod(indices, Fout) % (nx * kernel_size) // kernel_size
        iy = mod(indices, Fout) // (nx * kernel_size * kernel_size)
        itheta = indices // Fout

        index = ix + nx * iy + nx * ny * itheta

        return index.unsqueeze(1).long()

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): pooled tensor.
        """
        if self.shortcut:
            return x

        return self.expansion(x, self.index)


class SO3SpatialUnpool(nn.Module):
    """
    An hyper-icosahedral pooling layer - well-suited for SO(3) group.
    Pooling is based on the assumption the vertices are sorted the same way as Max Jiang.
    See: https://github.com/maxjiang93/ugscnn/blob/master/meshcnn/mesh.py
    """

    def __init__(self, kernel_size, size, expansion):
        """
        Args:
            kernel_size (int): pooling reduction in format (F_pool).
            size (sequence of ints): dimensions of the hyper-icosahedre in format (L, F).
            expansion (str): expansion operation.
        """

        super(SO3SpatialUnpool, self).__init__()

        if expansion not in {"max", "avg", "rand"}:
            raise ValueError(f"{expansion} is not a valid value for expansion, must be 'max' 'avg' or 'rand'.")

        self.index = self.get_expansion_index(size, kernel_size)

        if expansion == "rand":
            self.expansion = rand_unpool
        elif expansion == "max":
            self.expansion = max_unpool
        else:
            self.expansion = avg_unpool

        self.shortcut = kernel_size == 1

    def get_expansion_index(self, size, kernel_size):
        ns, nalpha = size

        lvl_in = int(math.log((ns - 2) / 10) / math.log(4))
        lvl_out = lvl_in + int(math.log(kernel_size) / math.log(2))

        if lvl_out - lvl_in > 1:
            raise NotImplementedError

        pkl_data = pkgutil.get_data(__name__, os.path.join("s2_upsampling", f"expansion_lvl{lvl_in}_lvl{lvl_out}.pt"))
        indices = torch.load(io.BytesIO(pkl_data))[None, :, :] + (torch.arange(nalpha) * ns)[:, None, None]

        return indices.reshape(-1, 2)

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): pooled tensor.
        """
        if self.shortcut:
            return x

        return self.expansion(x, self.index)
