# coding=utf-8

import math

import torch

from ..utils.utils import mod, weighted_norm


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
    x = G[..., 0, 2]
    y = G[..., 1, 2]
    theta = mod(torch.atan2(G[..., 1, 0], G[..., 0, 0]), math.pi, -math.pi / 2)
    return x, y, theta


def se2_inverse(G):
    """
    Returns a new tensor corresponding to the inverse of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): matrix formulation of the inverse group elements.
    """
    return torch.inverse(G)


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

    # transform group product to be sure the element is in the projective line bundle of the se2 group
    x, y, theta = se2_element(G)
    G = se2_matrix(x, y, theta)

    return weighted_norm(se2_log(G), Re)


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
    if nx == 1:
        x_axis = torch.zeros(1)
    else:
        x_axis = torch.arange(0.0, 1.0, 1 / nx)

    if ny == 1:
        y_axis = torch.zeros(1)
    else:
        y_axis = torch.arange(0.0, 1.0, 1 / ny)

    if ntheta == 1:
        theta_axis = torch.zeros(1)
    else:
        theta_axis = torch.arange(-math.pi / 2, math.pi / 2, math.pi / ntheta)

    theta, y, x = torch.meshgrid(theta_axis, y_axis, x_axis)

    sampling = torch.stack((x.flatten(), y.flatten(), theta.flatten()))

    return sampling
