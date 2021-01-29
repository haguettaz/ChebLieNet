import math
from typing import Optional, Tuple

import torch
from pykeops.torch import LazyTensor, Pm
from torch import FloatTensor
from torch import device as Device

from ..utils import mod


def se2_matrix(x: FloatTensor, y: FloatTensor, theta: FloatTensor, device: Device) -> FloatTensor:
    """
    Returns a new tensor corresponding to matrix formulation of the given input tensors representing
    SE(2) group elements.

    Args:
        x (FloatTensor): x translation input tensor.
        y (FloatTensor): y translation input tensor.
        theta (FloatTensor): z rotation input tensor.
        device (Device): computation device.

    Returns:
        FloatTensor: matrix representation output tensor.
    """
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    Gg = torch.zeros((x.shape[0], 3, 3), device=device)
    Gg[:, 0, 0] = cos
    Gg[:, 0, 1] = -sin
    Gg[:, 0, 2] = x
    Gg[:, 1, 0] = sin
    Gg[:, 1, 1] = cos
    Gg[:, 1, 2] = y
    Gg[:, 2, 2] = 1.0

    return Gg


def se2_log(Gg: LazyTensor) -> LazyTensor:
    """
    Returns a new LazyTensor corresponding to logarithmic maps of the matrix representation input LazyTensor.

    Args:
        Gg (LazyTensor): input LazyTensor, i.e. matrix representation.

    Returns:
        (LazyTensor): output LazyTensor, i.e. tangent space coefficients.
    """

    theta = Gg[0].atan2(Gg[3]).mod(math.pi, -math.pi / 2)
    x = Gg[2]
    y = Gg[5]

    c1 = theta / 2 * y + x * (theta / 2).cos() / ((theta / 2).sinxdivx())
    c2 = -theta / 2 * x + y * (theta / 2).cos() / ((theta / 2).sinxdivx())
    c3 = theta

    return LazyTensor.cat((c1, c2, c3), dim=-1)


def se2_anisotropic_square_riemannanian_distance(
    xi: LazyTensor, xj: LazyTensor, sigmas: Tuple[float, float, float], device: Device
) -> LazyTensor:
    """
    Returns the square anisotropic riemannian distances between xi and xj.

    Args:
        xi (LazyTensor): input tensor, i.e. source points in format (N, 1, 3).
        xj (LazyTensor): input tensor, i.e. target points in format (1, N, 3).
        sigmas (tuple): anisotropy's parameters to compute anisotropic riemannian distances.
        device (Device): computation device.

    Returns:
        (LazyTensor): output tensor, i.e. square anisotropic riemannian distances in format (N, N, 1).
    """
    S = Pm(torch.tensor([*sigmas], device=device))

    xixj = LazyTensor.keops_tensordot(xi, xj, (3, 3), (3, 3), (1,), (0,))  # g^-1 * h

    return se2_log(xixj).weightedsqnorm(S)