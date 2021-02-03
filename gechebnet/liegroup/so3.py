import math

import torch
from pykeops.torch import LazyTensor, Pm
from torch import device as Device

from ..utils import mod

ROUND_PI = 3.14159


def so3_matrix(alpha, beta, gamma, device=None):
    """
    Returns a new tensor corresponding to matrix formulation of the given input tensors representing
    SO(3) group elements.

    Args:
        alpha (FloatTensor): x rotation input tensor.
        beta (FloatTensor): y rotation input tensor.
        gamma (FloatTensor): z rotation input tensor.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        FloatTensor: matrix representation output tensor.
    """
    cosa = torch.cos(alpha)
    sina = torch.sin(alpha)
    Ra = torch.zeros((alpha.shape[0], 3, 3), device=device)
    Ra[:, 1, 1] = cosa
    Ra[:, 1, 2] = -sina
    Ra[:, 2, 1] = sina
    Ra[:, 2, 2] = cosa
    Ra[:, 0, 0] = 1.0

    cosb = torch.cos(beta)
    sinb = torch.sin(beta)
    Rb = torch.zeros((beta.shape[0], 3, 3), device=device)
    Rb[:, 0, 0] = cosb
    Rb[:, 0, 2] = sinb
    Rb[:, 2, 0] = -sinb
    Rb[:, 2, 2] = cosb
    Rb[:, 1, 1] = 1.0

    cosc = torch.cos(gamma)
    sinc = torch.sin(gamma)
    Rc = torch.zeros((gamma.shape[0], 3, 3), device=device)
    Rc[:, 0, 0] = cosc
    Rc[:, 0, 1] = -sinc
    Rc[:, 1, 0] = sinc
    Rc[:, 1, 1] = cosc
    Rc[:, 2, 2] = 1.0

    return Rc @ Rb @ Ra


def so3_log(Gg: LazyTensor, Gg_t: LazyTensor):
    """
    Returns a new LazyTensor corresponding to logarithmic maps of the matrix representation input LazyTensor.
    The matrix logarithmic is computed using Rodrigues' rotation formula.

    Args:
        Gg (LazyTensor): input LazyTensor, i.e. matrix representation.
        Gg_t (LazyTensor): input LazyTensor, i.e. transposed matrix representation.

    Returns:
        (LazyTensor): output LazyTensor, i.e. tangent space coefficients.
    """
    # theta is a rounded to 5 decimal places's angle in the range [0.00000, 3.14159]
    theta = (((Gg[0] + Gg[4] + Gg[8]) - 1) / 2).acos().round(5)

    A = 0.5 * (Gg - Gg_t) / theta.sinc() # implement transpose on PyKeops??

    # mimic if theta == pi then (0, 0, theta) else (a02, a10, a32) with step function
    c1 = 0.0 * (theta - ROUND_PI).step() + A[2] * (1.0 - (theta - ROUND_PI).step())
    c2 = 0.0 * (theta - ROUND_PI).step() + A[3] * (1.0 - (theta - ROUND_PI).step())
    c3 = theta * (theta - ROUND_PI).step() + A[7] * (1.0 - (theta - ROUND_PI).step())

    return LazyTensor.cat((c1, c2, c3), dim=-1)


def so3_anisotropic_square_riemannanian_distance(xi, xj, xi_t, xj_t, sigmas, device):
    """
    Returns the square anisotropic riemannian distances between xi and xj.

    Args:
        xi (LazyTensor): input tensor, i.e. source points in format (N, 1, 3).
        xj (LazyTensor): input tensor, i.e. target points in format (1, N, 3).
        xi_t (LazyTensor): input tensor, i.e. source points in format (N, 1, 3) for the transposed operation.
        xj_t (LazyTensor): input tensor, i.e. target points in format (1, N, 3) for the transposed operation.
        sigmas (tuple): anisotropy's parameters to compute anisotropic riemannian distances.
        device (Device): computation device.

    Returns:
        (LazyTensor): output tensor, i.e. square anisotropic riemannian distances in format (N, N, 1).
    """
    S = Pm(torch.tensor([*sigmas], device=device))

    xixj = LazyTensor.keops_tensordot(xi, xj, (3, 3), (3, 3), (1,), (0,))  # Gg^{-1}.Gh
    xjxi_t = LazyTensor.keops_tensordot(xj_t, xi_t, (3, 3), (3, 3), (1,), (0,))  # Gh{-1}.Gg

    return so3_log(xixj, xjxi_t).weightedsqnorm(S)
