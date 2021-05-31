# coding=utf-8

import io
import math
import os
import pkgutil

import torch
from torch import nn

from ...utils.utils import mod


def avg_pool(x, index=None):
    if index is None:
        return x.mean(dim=-1)
    return x[..., index].mean(dim=-1)


def max_pool(x, index=None):
    if index is None:
        out, _ = x.max(dim=-1)
        return out
    out, _ = x[..., index].max(dim=-1)
    return out


def rand_pool(x, index=None):
    if index is None:
        raise NotImplementedError
    N, K = index.shape
    out = x[..., index[torch.arange(N), torch.randint(K, (N,))]]
    return out


class SE2SpatialPool(nn.Module):
    """
    A SE(2) spatial pooling layer. Required the  to be ordered L, H, W.
    """

    def __init__(self, kernel_size, size, reduction):
        """
        Args:
            kernel_size (int): pooling reduction size.
            size (sequence of ints): size in format (nx, ny, ntheta)
            reduction (str): reduction operation.
        """
        super(SE2SpatialPool, self).__init__()

        if reduction not in {"max", "avg", "rand"}:
            raise ValueError(f"{reduction} is not a valid value for reduction, must be 'max' 'avg' or 'rand'.")

        self.index = self.get_reduction_index(size, kernel_size)

        if reduction == "rand":
            self.reduction = rand_pool
        elif reduction == "max":
            self.reduction = max_pool
        else:
            self.reduction = avg_pool

        self.shortcut = kernel_size == 1

    def get_reduction_index(self, size, kernel_size):
        nx, ny, ntheta = size

        Fin = nx * ny
        Fout = nx // kernel_size * ny // kernel_size

        x_idx = torch.arange(nx // kernel_size)
        y_idx = torch.arange(ny // kernel_size)

        grid_y, grid_x = torch.meshgrid(y_idx, x_idx)

        x_idx, y_idx = grid_x.flatten(), grid_y.flatten()

        theta_index = torch.arange(ntheta)
        xy_index = torch.empty((Fout, kernel_size ** 2))
        for ikx in range(kernel_size):
            for iky in range(kernel_size):
                xy_index[:, ikx + iky * kernel_size] = x_idx * kernel_size + y_idx * kernel_size * nx + ikx + iky * nx

        index = theta_index[:, None, None] * Fin + xy_index[None, :, :]
        return index.reshape(-1, kernel_size ** 2).long()

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): pooled tensor.
        """
        if self.shortcut:
            return x

        return self.reduction(x, self.index)


class SO3SpatialPool(nn.Module):
    """
    An hyper-icosahedral pooling layer - well-suited for SO(3) group.
    Pooling is based on the assumption the vertices are sorted the same way as Max Jiang.
    See: https://github.com/maxjiang93/ugscnn/blob/master/meshcnn/mesh.py
    """

    def __init__(self, kernel_size, size, reduction):
        """
        Args:
            kernel_size (int): pooling reduction size.
            size (sequence of ints): dimensions of the hyper-icosahedre in format (L, F).
        """

        super(SO3SpatialPool, self).__init__()

        if reduction not in {"max", "avg", "rand"}:
            raise ValueError(f"{reduction} is not a valid value for reduction, must be 'max' 'avg' or 'rand'.")

        self.index = self.get_reduction_index(size, kernel_size)

        if reduction == "rand":
            self.reduction = rand_pool
        elif reduction == "max":
            self.reduction = max_pool
        else:
            self.reduction = avg_pool

        self.shortcut = kernel_size == 1

    def get_reduction_index(self, size, kernel_size):
        ns, nalpha = size

        lvl_in = int(math.log((ns - 2) / 10) / math.log(4))
        lvl_out = lvl_in - int(math.log(kernel_size) / math.log(2))

        if lvl_in - lvl_out > 1:
            raise NotImplementedError

        pkl_data = pkgutil.get_data(__name__, os.path.join("s2_downsampling", f"reduction_lvl{lvl_in}_lvl{lvl_out}.pt"))
        indices = torch.load(io.BytesIO(pkl_data))[None, :, :] + (torch.arange(nalpha) * ns)[:, None, None]

        return indices.reshape(-1, 7)

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): pooled tensor.
        """
        if self.shortcut:
            return x

        return self.reduction(x, self.index)


class GlobalPool(nn.Module):
    """
    A global pooling layer, on spatial and orientation's dimensions.
    """

    def __init__(self, reduction):
        """
        Args:
            reduction (int): size of the graph, that is its number of .
        """

        super(GlobalPool, self).__init__()

        if reduction not in {"max", "avg", "rand"}:
            raise ValueError(f"{reduction} is not a valid value for reduction, must be 'max' 'avg' or 'rand'.")

        if reduction == "rand":
            self.reduction = rand_pool
        elif reduction == "max":
            self.reduction = max_pool
        else:
            self.reduction = avg_pool

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): pooled tensor.
        """

        return self.reduction(x)
