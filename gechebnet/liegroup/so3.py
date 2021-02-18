import math
from typing import Optional, Tuple

import torch
#from pykeops.torch import LazyTensor, Pm
from torch import FloatTensor, Tensor
from torch import device as Device

from ..utils import mod
from .utils import weighted_norm

# ROUND_PI = 3.14159


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
    if not alpha.nelement() == beta.nelement() == gamma.nelement():
        raise ValueError(f"input tensors must contain the same number of element but do not.")

    N = alpha.nelement()

    cosa = torch.cos(alpha)
    sina = torch.sin(alpha)
    Ra = torch.zeros((N, 3, 3), device=device)
    Ra[:, 1, 1] = cosa
    Ra[:, 1, 2] = -sina
    Ra[:, 2, 1] = sina
    Ra[:, 2, 2] = cosa
    Ra[:, 0, 0] = 1.0

    cosb = torch.cos(beta)
    sinb = torch.sin(beta)
    Rb = torch.zeros((N, 3, 3), device=device)
    Rb[:, 0, 0] = cosb
    Rb[:, 0, 2] = sinb
    Rb[:, 2, 0] = -sinb
    Rb[:, 2, 2] = cosb
    Rb[:, 1, 1] = 1.0

    cosc = torch.cos(gamma)
    sinc = torch.sin(gamma)
    Rc = torch.zeros((N, 3, 3), device=device)
    Rc[:, 0, 0] = cosc
    Rc[:, 0, 1] = -sinc
    Rc[:, 1, 0] = sinc
    Rc[:, 1, 1] = cosc
    Rc[:, 2, 2] = 1.0

    return Rc @ Rb @ Ra


# def so3_log(Gg: LazyTensor):
#     """
#     Returns a new LazyTensor corresponding to logarithmic maps of the matrix representation input LazyTensor.
#     The matrix logarithmic is computed using Rodrigues' rotation formula.

#     Args:
#         Gg (LazyTensor): input LazyTensor, i.e. matrix representation.
#         Gg_t (LazyTensor): input LazyTensor, i.e. transposed matrix representation.

#     Returns:
#         (LazyTensor): output LazyTensor, i.e. tangent space coefficients.
#     """
#     # theta is a rounded to 5 decimal places's angle in the range [0.00000, 3.14159]
#     theta = (((Gg[0] + Gg[4] + Gg[8]) - 1) / 2).acos().round(5)

#     c1 = 0.5 * (Gg[2] - Gg[6]) / theta.sinc()
#     c2 = 0.5 * (Gg[3] - Gg[1]) / theta.sinc()
#     c3 = 0.5 * (Gg[7] - Gg[5]) / theta.sinc()

#     # mimic if theta == pi then (0, 0, theta) else (a02, a10, a32) with step function
#     c1 = 0.0 * (theta - ROUND_PI).step() + c1 * (1.0 - (theta - ROUND_PI).step())
#     c2 = 0.0 * (theta - ROUND_PI).step() + c2 * (1.0 - (theta - ROUND_PI).step())
#     c3 = theta * (theta - ROUND_PI).step() + c3 * (1.0 - (theta - ROUND_PI).step())

#     return LazyTensor.cat((c1, c2, c3), dim=-1)


def so3_inverse(G):
    return torch.transpose(G, -1, -2)


def so3_element(G):
    alpha = mod(torch.atan2(G[..., 2, 1], G[..., 2, 2]), math.pi, -math.pi / 2)
    gamma = mod(torch.atan2(G[..., 1, 0], G[..., 0, 0]), math.pi, -math.pi / 2)
    beta = torch.atan2(-G[..., 2, 0], G[..., 0, 0] / torch.cos(gamma))
    return alpha, beta, gamma


def so3_log(G):
    theta = torch.acos(((G[..., 0, 0] + G[..., 1, 1] + G[..., 2, 2]) - 1) / 2)  # round??

    c1 = 0.5 * theta / torch.sin(theta) * (G[..., 0, 2] - G[..., 2, 0])
    c2 = 0.5 * theta / torch.sin(theta) * (G[..., 1, 0] - G[..., 0, 1])
    c3 = 0.5 * theta / torch.sin(theta) * (G[..., 2, 1] - G[..., 1, 2])

    mask = theta == 0.0
    c1[mask] = 0.5 * G[mask, 0, 2] - G[mask, 2, 0]
    c2[mask] = 0.5 * G[mask, 1, 0] - G[mask, 0, 1]
    c3[mask] = 0.5 * G[mask, 2, 1] - G[mask, 1, 2]

    mask = theta == math.pi
    c1[mask] = 0.0
    c2[mask] = 0.0
    c3[mask] = 3.14159

    c = torch.stack((c1, c2, c3), dim=-1).unsqueeze(2)

    return c


def so3_riemannian_sqdist(Gg, Gh, Re):
    G = torch.matmul(so3_inverse(Gg), Gh)

    # transform group product to be sure the element is in the projective line bundle of the so3 group
    alpha_, beta_, gamma_ = so3_element(G)
    G = so3_matrix(alpha_, beta_, gamma_)

    return weighted_norm(so3_log(G), Re)


# def so3_anisotropic_square_riemannanian_distance(xi, xj, sigmas):
#     """
#     Returns the square anisotropic riemannian distances between xi and xj.

#     Args:
#         xi (LazyTensor): input tensor, i.e. source points in format (N, 1, 3).
#         xj (LazyTensor): input tensor, i.e. target points in format (1, N, 3).
#         xi_t (LazyTensor): input tensor, i.e. source points in format (N, 1, 3) for the transposed operation.
#         xj_t (LazyTensor): input tensor, i.e. target points in format (1, N, 3) for the transposed operation.
#         sigmas (tuple): anisotropy's parameters to compute anisotropic riemannian distances.
#         device (Device): computation device.

#     Returns:
#         (LazyTensor): output tensor, i.e. square anisotropic riemannian distances in format (N, N, 1).
#     """
#     S = Pm(torch.tensor([*sigmas]))

#     xixj = LazyTensor.keops_tensordot(xi, xj, (3, 3), (3, 3), (1,), (0,))  # Gg^{-1}.Gh

#     return so3_log(xixj).weightedsqnorm(S)


def xyz2betagamma(x: FloatTensor, y: FloatTensor, z: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
    """
    Returns new tensors corresponding to angle representation from the cartesian representation.

    Args:
        x (FloatTensor): input tensor, i.e. x positions.
        y (FloatTensor): input tensor, i.e. y positions.
        z (FloatTensor): input tensor, i.e. z positions.

    Returns:
        (FloatTensor): output tensor, i.e. beta rotation about y axis.
        (FloatTensor): output tensor, i.e. gamma rotation about z axis.
    """

    beta = torch.stack(
        (
            torch.atan2(-z, -torch.sqrt(x.pow(2) + y.pow(2))),
            torch.atan2(-z, torch.sqrt(x.pow(2) + y.pow(2))),
        ),
        dim=-1,
    )

    gamma = torch.stack((torch.atan2(-y, -x), torch.atan2(y, x)), dim=-1)

    mask = (beta >= -math.pi) & (beta < math.pi) & (gamma >= -math.pi / 2) & (gamma < math.pi / 2)

    return beta[mask], gamma[mask]


def alphabetagamma2xyz(
    alpha: FloatTensor, beta: FloatTensor, gamma: FloatTensor, axis=None
) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
    """
    Returns new tensors corresponding to angle representation from the cartesian representation.

    Args:
        alpha (FloatTensor): input tensor, i.e. alpha rotation about x axis.
        beta (FloatTensor): input tensor, i.e. beta rotation about y axis.
        gamma (FloatTensor): input tensor, i.e. gamma rotation about z axis.

    Returns:
        (FloatTensor): output tensor, i.e. x positions.
        (FloatTensor): output tensor, i.e. y positions.
        (FloatTensor): output tensor, i.e. z positions.
    """
    if axis == "x":
        return (math.pi + alpha) * torch.cos(beta) * torch.cos(gamma)
    if axis == "y":
        return (math.pi + alpha) * torch.cos(beta) * torch.sin(gamma)
    if axis == "z":
        return -(math.pi + alpha) * torch.sin(beta)

    x = (math.pi + alpha) * torch.cos(beta) * torch.cos(gamma)
    y = (math.pi + alpha) * torch.cos(beta) * torch.sin(gamma)
    z = -(math.pi + alpha) * torch.sin(beta)

    return x, y, z
