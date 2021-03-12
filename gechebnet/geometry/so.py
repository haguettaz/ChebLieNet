# coding=utf-8

import math
import os
import pickle
import pkgutil

import torch

from ..utils.utils import mod, weighted_norm
from .utils import rotation_matrix, xyz2betagamma


def s2_uniform_sampling(num_samples):
    """
    Uniformly samples elements of S(2), the 1-sphere.

    Args:
        num_samples (int): level of the icosahedral sampling. Level 0 corresponds to an icosahedre.

    Returns:
        (`torch.FloatTensor`): beta coordinates of the uniform sampling.
        (`torch.FloatTensor`): gamma coordinates of the uniform sampling.
    """

    level = math.log((num_samples - 2) / 10) / math.log(4)
    if not level.is_integer():
        raise ValueError(f"{num_samples} is not a valid value for argument num_samples...")

    pkl_data = pkgutil.get_data(__name__, os.path.join("s2_sampling", f"icosphere_{int(level)}.pkl"))
    data = pickle.loads(pkl_data)
    xyz = torch.from_numpy(data["V"])

    # project xyz on the 1-sphere, just to be sure...
    xyz /= xyz.norm(dim=1, keepdim=True)

    beta, gamma = xyz2betagamma(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return beta, gamma


def so3_uniform_sampling(level, nalpha):
    """
    Uniformly samples elements of the SE(2) group in the hypersphere S(2) x [-pi/2, pi/2).

    Args:
        level (int): level of the icosahedral sampling. Level 0 corresponds to an icosahedre.
        naplha (int): discretization of the alpha axis.

    Returns:
        (`torch.FloatTensor`): alpha coordinates of the uniform sampling.
        (`torch.FloatTensor`): beta coordinates of the uniform sampling.
        (`torch.FloatTensor`): gamma coordinates of the uniform sampling.
    """
    beta, gamma = s2_uniform_sampling(level)
    N = beta.size(0)

    # uniformly samples alpha
    if nalpha < 2:
        raise ValueError(f"Cannot sample element on SO3 with nalpha < 2. Use `s2_uniform_sampling` instead.")

    alpha = torch.arange(-math.pi / 2, math.pi / 2, math.pi / nalpha)

    return (
        alpha.unsqueeze(1).expand(nalpha, N).flatten(),
        beta.unsqueeze(0).expand(nalpha, N).flatten(),
        gamma.unsqueeze(0).expand(nalpha, N).flatten(),
    )


def s2_matrix(beta, gamma, device=None):
    """
    Returns a new tensor corresponding to matrix formulation of the given input tensors representing
    SO(3) group elements.

    Args:
        beta (`torch.FloatTensor`): beta attributes of group elements.
        gamma (`torch.FloatTensor`): gamma attributes of group elements.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        (`torch.FloatTensor`): matrix representation of the group elements.
    """
    R_beta_y = rotation_matrix(beta, "y", device)
    R_gamma_z = rotation_matrix(gamma, "z", device)
    return R_gamma_z @ R_beta_y


def so3_matrix(alpha, beta, gamma, device=None):
    """
    Returns a new tensor corresponding to matrix formulation of the given input tensors representing
    SO(3) group elements.

    Args:
        alpha (`torch.FloatTensor`): alpha attributes of group elements.
        beta (`torch.FloatTensor`): beta attributes of group elements.
        gamma (`torch.FloatTensor`): gamma attributes of group elements.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        (`torch.FloatTensor`): matrix representation of the group elements.
    """
    R_alpha_z = rotation_matrix(alpha, "z", device)
    R_beta_y = rotation_matrix(beta, "y", device)
    R_gamma_z = rotation_matrix(gamma, "z", device)
    return R_gamma_z @ R_beta_y @ R_alpha_z


def s2_inverse(G):
    """
    Returns a new tensor corresponding to the inverse of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): matrix formulation of the inverse group elements.
    """
    return torch.transpose(G, -1, -2)


def so3_inverse(G):
    """
    Returns a new tensor corresponding to the inverse of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): matrix formulation of the inverse group elements.
    """
    return torch.transpose(G, -1, -2)


def s2_element(G):
    """
    Return new tensors corresponding to alpha, beta and gamma attributes of the group elements specified by the
    S2 group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): beta attributes of the group elements.
        (`torch.FloatTensor`): gamma attributes of the group elements.
    """

    gamma = torch.atan2(G[..., 1, 0], G[..., 0, 0])
    beta = torch.acos(G[..., 2, 2])
    return beta, gamma


def so3_element(G):
    """
    Return new tensors corresponding to alpha, beta and gamma attributes of the group elements specified by the
    so3 group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): alpha attributes of the group elements.
        (`torch.FloatTensor`): beta attributes of the group elements.
        (`torch.FloatTensor`): gamma attributes of the group elements.
    """

    gamma = torch.atan2(G[..., 1, 2], G[..., 0, 2])
    sin = torch.sin(gamma)
    cos = torch.cos(gamma)

    beta = torch.atan2(cos * G[..., 0, 2] + sin * G[..., 1, 2], G[..., 2, 2])
    alpha = torch.atan2(-sin * G[..., 0, 0] + cos * G[..., 1, 0], -sin * G[..., 0, 1] + cos * G[..., 1, 1])

    return alpha, beta, gamma


def s2_log(G):
    """
    Returns a new tensor containing the riemannnian logarithm of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): riemannian logarithms.
    """

    beta, gamma = s2_element(G)
    G = so3_matrix(-gamma, beta, gamma)

    theta = torch.acos(((G[..., 0, 0] + G[..., 1, 1] + G[..., 2, 2]) - 1) / 2)

    c1 = 0.5 * theta / torch.sin(theta) * (G[..., 2, 1] - G[..., 1, 2])
    c2 = 0.5 * theta / torch.sin(theta) * (G[..., 0, 2] - G[..., 2, 0])
    c3 = torch.zeros_like(c1)

    mask = theta == 0.0
    c1[mask] = 0.5 * G[mask, 2, 1] - G[mask, 1, 2]
    c2[mask] = 0.5 * G[mask, 0, 2] - G[mask, 2, 0]

    mask = theta == math.pi
    c1[mask] = math.pi
    c2[mask] = 0.0

    c = torch.stack((c1, c2, c3), dim=-1).unsqueeze(2)

    return c


def so3_log(G):
    """
    Returns a new tensor containing the riemannnian logarithm of the group elements in matrix formulation.

    Args:
        G (`torch.FloatTensor`): matrix formulation of the group elements.

    Returns:
        (`torch.FloatTensor`): riemannian logarithms.
    """
    theta = torch.acos(((G[..., 0, 0] + G[..., 1, 1] + G[..., 2, 2]) - 1) / 2)

    c1 = 0.5 * theta / torch.sin(theta) * (G[..., 2, 1] - G[..., 1, 2])
    c2 = 0.5 * theta / torch.sin(theta) * (G[..., 0, 2] - G[..., 2, 0])
    c3 = 0.5 * theta / torch.sin(theta) * (G[..., 1, 0] - G[..., 0, 1])

    mask = theta == 0.0
    c1[mask] = 0.5 * G[mask, 2, 1] - G[mask, 1, 2]
    c2[mask] = 0.5 * G[mask, 0, 2] - G[mask, 2, 0]
    c3[mask] = 0.5 * G[mask, 1, 0] - G[mask, 0, 1]

    mask = theta == math.pi
    c1[mask] = math.pi
    c2[mask] = 0.0
    c3[mask] = 0.0

    c = torch.stack((c1, c2, c3), dim=-1).unsqueeze(2)

    return c


def s2_riemannian_sqdist(Gg, Gh, Re):
    """
    Return the squared riemannian distances between group elements in matrix formulation.

    Args:
        Gg (`torch.FloatTensor`): matrix formulation of the source group elements.
        Gh (`torch.FloatTensor`): matrix formulation of the target group elements.
        Re (`torch.FloatTensor`): matrix formulation of the riemannian metric.

    Returns:
        (`torch.FloatTensor`): squared riemannian distances
    """
    G = torch.matmul(s2_inverse(Gg), Gh)

    return weighted_norm(s2_log(G), Re)


def so3_riemannian_sqdist(Gg, Gh, Re):
    """
    Returns the squared riemannian distances between group elements in matrix formulation.

    Args:
        Gg (`torch.FloatTensor`): matrix formulation of the source group elements.
        Gh (`torch.FloatTensor`): matrix formulation of the target group elements.
        Re (`torch.FloatTensor`): matrix formulation of the riemannian metric.

    Returns:
        (`torch.FloatTensor`): squared riemannian distances
    """
    G = torch.matmul(so3_inverse(Gg), Gh)

    alpha, beta, gamma = so3_element(G)

    sqdist1 = weighted_norm(so3_log(so3_matrix(alpha, beta, gamma)), Re)
    sqdist2 = weighted_norm(so3_log(so3_matrix(alpha - math.pi, beta, gamma)), Re)
    sqdist3 = weighted_norm(so3_log(so3_matrix(alpha + math.pi, beta, gamma)), Re)

    sqdist, _ = torch.stack((sqdist1, sqdist2, sqdist3)).min(dim=0)

    return sqdist
