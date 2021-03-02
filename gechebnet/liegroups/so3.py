import math
import os
import pickle
from typing import Optional, Tuple

import torch
from torch import FloatTensor, Tensor
from torch import device as Device

from ..utils.polyhedron import SphericalPolyhedron
from ..utils.utils import mod, weighted_norm


def so3_matrix(alpha, beta, gamma, device=None):
    """
    Returns a new tensor corresponding to matrix formulation of the given input tensors representing
    SO(3) group elements.

    Args:
        alpha (FloatTensor): alpha attributes of group elements.
        beta (FloatTensor): beta attributes of group elements.
        gamma (FloatTensor): gamma attributes of group elements.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        (FloatTensor): matrix representation of the group elements.
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


def so3_element(G):
    """
    Returns three new tensors corresponding to alpha, beta and gamma attributes of the group elements specified by the
    so3 group elements in matrix formulation.

    Args:
        G (Tensor): matrix formulation of the group elements.

    Returns:
        (Tensor): alpha attributes of the group elements.
        (Tensor): beta attributes of the group elements.
        (Tensor): gamma attributes of the group elements.
    """
    alpha = mod(torch.atan2(G[..., 2, 1], G[..., 2, 2]), math.pi, -math.pi / 2)
    gamma = mod(torch.atan2(G[..., 1, 0], G[..., 0, 0]), math.pi, -math.pi / 2)
    beta = torch.atan2(-G[..., 2, 0], G[..., 0, 0] / torch.cos(gamma))
    return alpha, beta, gamma


def so3_inverse(G):
    """
    Returns a new tensor corresponding to the inverse of the group elements in matrix formulation.

    Args:
        G (Tensor): matrix formulation of the group elements.

    Returns:
        (Tensor): matrix formulation of the inverse group elements.
    """
    return torch.transpose(G, -1, -2)


def so3_log(G):
    """
    Returns a new tensor containing the riemannnian logarithm of the group elements in matrix formulation.

    Args:
        G (Tensor): matrix formulation of the group elements.

    Returns:
        (Tensor): riemannian logarithms.
    """
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
    """
    Returns the squared riemannian distances between group elements in matrix formulation.

    Args:
        Gg (Tensor): matrix formulation of the source group elements.
        Gh (Tensor): matrix formulation of the target group elements.
        Re (Tensor): matrix formulation of the riemannian metric.

    Returns:
        (Tensor): squared riemannian distances
    """
    G = torch.matmul(so3_inverse(Gg), Gh)

    # transform group product to be sure the element is in the projective line bundle of the so3 group
    alpha_, beta_, gamma_ = so3_element(G)
    G = so3_matrix(alpha_, beta_, gamma_)

    return weighted_norm(so3_log(G), Re)


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


def so3_uniform_sampling(path_to_sampling, level, nalpha):
    with open(os.path.join(path_to_sampling, f"icosphere_{level}.pkl"), "rb") as f:
        data = pickle.load(f)
        xyz = torch.from_numpy(data["V"])

    beta, gamma = xyz2betagamma(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    N = beta.size(0)

    # uniformly samples alpha
    if nalpha == 1:
        alpha = torch.zeros(1)
    else:
        alpha = torch.arange(-math.pi / 2, math.pi / 2, math.pi / nalpha)

    # merge sampling
    sampling = torch.stack(
        (
            alpha.unsqueeze(1).expand(nalpha, N).flatten(),
            beta.unsqueeze(0).expand(nalpha, N).flatten(),
            gamma.unsqueeze(0).expand(nalpha, N).flatten(),
        )
    )

    return sampling
