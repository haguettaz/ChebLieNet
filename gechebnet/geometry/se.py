# coding=utf-8

import math
import os
import pickle

import torch

from ..utils.utils import mod, weighted_norm
from .utils import rotation_matrix, xyz2betagamma


def r2_uniform_sampling(nx, ny):
    """
    Uniformly samples elements of the SE(2) group in the hypercube [0, 1) x [0, 1) x [-pi/2, pi/2).

    Args:
        nx (int): discretization of the x axis.
        ny (int): discretization of the y axis.

    Returns:
        (`torch.FloatTensor`): uniform sampling.
    """
    if nx < 2 or ny < 2:
        raise ValueError(f"resolution of x and y axis must be greater than 2.")

    y, x = torch.meshgrid(torch.arange(0.0, 1.0, 1 / ny), torch.arange(0.0, 1.0, 1 / nx))
    return x.flatten(), y.flatten()


def se2_uniform_sampling(nx, ny, ntheta):
    """
    Uniformly samples elements of the SE(2) group in the hypercube [0, 1) x [0, 1) x [-pi/2, pi/2).

    Args:
        nx (int): discretization of the x axis.
        ny (int): discretization of the y axis.
        ntheta (int): discretization of the theta axis.

    Returns:
        (`torch.FloatTensor`): uniform sampling.
    """

    x, y = r2_uniform_sampling(nx, ny)

    # uniformly samples alpha
    if ntheta < 2:
        raise ValueError(f"Cannot sample element on SE2 with nalpha < 2. Use `r2_uniform_sampling` instead.")

    theta = torch.arange(-math.pi / 2, math.pi / 2, math.pi / ntheta)

    return (
        x.unsqueeze(0).expand(ntheta, nx * ny).flatten(),
        y.unsqueeze(0).expand(ntheta, nx * ny).flatten(),
        theta.unsqueeze(1).expand(ntheta, nx * ny).flatten(),
    )


def r2_matrix(x, y, device=None):
    """
    Returns a new tensor corresponding to matrix formulation of the given input tensors representing
    R(2) group elements.

    Args:
        x (`torch.FloatTensor`): x attributes of group elements.
        y (`torch.FloatTensor`): y attributes of group elements.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        (`torch.FloatTensor`): matrix representation of the group elements.
    """
    G = torch.zeros((x.nelement(), 3, 3), device=device)
    G[..., 0, 2] = x
    G[..., 1, 2] = y
    G[..., 0, 0] = 1.0
    G[..., 1, 1] = 1.0
    G[..., 2, 2] = 1.0

    return G


def se2_matrix(x, y, theta, device=None):
    """
    Returns a new tensor corresponding to matrix formulation of the given input tensors representing
    SE(2) group elements.

    Args:
        x (`torch.FloatTensor`): x attributes of group elements.
        y (`torch.FloatTensor`): y attributes of group elements.
        theta (`torch.FloatTensor`): theta attributes of group elements.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        (`torch.FloatTensor`): matrix representation of the group elements.
    """

    G = rotation_matrix(theta, "z", device=device)
    G[:, 0, 2] = x
    G[:, 1, 2] = y

    return G


def r2_inverse(G):
    """
    Returns a new tensor corresponding to the inverse of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): matrix formulation of the inverse group elements.
    """
    return torch.inverse(G)


def se2_inverse(G):
    """
    Returns a new tensor corresponding to the inverse of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): matrix formulation of the inverse group elements.
    """
    return torch.inverse(G)


def r2_element(G):
    """
    Return new tensors corresponding to alpha, beta and gamma attributes of the group elements specified by the
    S2 group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): beta attributes of the group elements.
        (`torch.FloatTensor`): gamma attributes of the group elements.
    """
    return G[..., 0, 2], G[..., 1, 2]


def se2_element(G):
    """
    Returns three new tensors corresponding to x, y and theta attributes of the group elements specified by the
    se2 group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): x attributes of the group elements.
        (`torch.FloatTensor`): y attributes of the group elements.
        (`torch.FloatTensor`): theta attributes of the group elements.
    """
    return G[..., 0, 2], G[..., 1, 2], torch.atan2(G[..., 1, 0], G[..., 0, 0])


def r2_log(G):
    """
    Returns a new tensor containing the riemannnian logarithm of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): riemannian logarithms.
    """
    x, y = r2_element(G)

    c1 = x
    c2 = y
    c3 = torch.zeros_like(c1)

    c = torch.stack((c1, c2, c3), dim=-1).unsqueeze(2)

    return c


def se2_log(G):
    """
    Returns a new tensor containing the riemannnian logarithm of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): riemannian logarithms.
    """
    x, y, theta = se2_element(G)

    c1 = theta / 2 * (y + x * torch.cos(theta / 2) / torch.sin(theta / 2))
    c2 = -theta / 2 * (x + y * torch.cos(theta / 2) / torch.sin(theta / 2))
    c3 = theta.clone()

    mask = theta == 0.0
    c1[mask] = x[mask]
    c2[mask] = y[mask]

    c = torch.stack((c1, c2, c3), dim=-1).unsqueeze(2)

    return c


def r2_riemannian_sqdist(Gg, Gh, Re):
    """
    Return the squared riemannian distances between group elements in matrix formulation.

    Args:
        Gg (`torch.FloatTensor`): matrix formulation of the source group elements.
        Gh (`torch.FloatTensor`): matrix formulation of the target group elements.
        Re (`torch.FloatTensor`): matrix formulation of the riemannian metric.

    Returns:
        (`torch.FloatTensor`): squared riemannian distances
    """
    G = torch.matmul(r2_inverse(Gg), Gh)

    return weighted_norm(r2_log(G), Re)


def se2_riemannian_sqdist(Gg, Gh, Re):
    """
    Returns the squared riemannian distances between group elements in matrix formulation.

    Args:
        Gg (`torch.FloatTensor`): matrix formulation of the source group elements.
        Gh (`torch.FloatTensor`): matrix formulation of the target group elements.
        Re (`torch.FloatTensor`): matrix formulation of the riemannian metric.

    Returns:
        (`torch.FloatTensor`): squared riemannian distances
    """
    G = torch.matmul(se2_inverse(Gg), Gh)
    x, y, theta = se2_element(G)

    # due to the symmetry of the anisotropic kernel, there is a pi-periodicity on the theta axis
    sqdist1 = weighted_norm(se2_log(se2_matrix(x, y, theta)), Re)
    sqdist2 = weighted_norm(se2_log(se2_matrix(x, y, theta - math.pi)), Re)
    sqdist3 = weighted_norm(se2_log(se2_matrix(x, y, theta + math.pi)), Re)

    sqdist, _ = torch.stack((sqdist1, sqdist2, sqdist3)).min(dim=0)

    return sqdist
