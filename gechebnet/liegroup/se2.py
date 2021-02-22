import math
from typing import Optional, Tuple

import torch
from torch import FloatTensor, Tensor
from torch import device as Device

from ..utils import mod, weighted_norm


def se2_matrix(x: FloatTensor, y: FloatTensor, theta: FloatTensor, device: Optional[Device] = None) -> FloatTensor:
    """
    Returns a new tensor corresponding to matrix formulation of the given input tensors representing
    SE(2) group elements.

    Args:
        x (FloatTensor): x attributes of group elements.
        y (FloatTensor): y attributes of group elements.
        theta (FloatTensor): theta attributes of group elements.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        (FloatTensor): matrix representation of the group elements.
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


def se2_element(G: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns three new tensors corresponding to x, y and theta attributes of the group elements specified by the
    se2 group elements in matrix formulation.

    Args:
        G (Tensor): matrix formulation of the group elements.

    Returns:
        (Tensor): x attributes of the group elements.
        (Tensor): y attributes of the group elements.
        (Tensor): theta attributes of the group elements.
    """
    x = G[..., 0, 2]
    y = G[..., 1, 2]
    theta = mod(torch.atan2(G[..., 1, 0], G[..., 0, 0]), math.pi, -math.pi / 2)
    return x, y, theta


def se2_inverse(G: Tensor) -> Tensor:
    """
    Returns a new tensor corresponding to the inverse of the group elements in matrix formulation.

    Args:
        G (Tensor): matrix formulation of the group elements.

    Returns:
        (Tensor): matrix formulation of the inverse group elements.
    """
    return torch.inverse(G)


def se2_log(G: Tensor) -> Tensor:
    """
    Returns a new tensor containing the riemannnian logarithm of the group elements in matrix formulation.

    Args:
        G (Tensor): matrix formulation of the group elements.

    Returns:
        (Tensor): riemannian logarithms.
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


def se2_riemannian_sqdist(Gg: Tensor, Gh: Tensor, Re: Tensor) -> Tensor:
    """
    Returns the squared riemannian distances between group elements in matrix formulation.

    Args:
        Gg (Tensor): matrix formulation of the source group elements.
        Gh (Tensor): matrix formulation of the target group elements.
        Re (Tensor): matrix formulation of the riemannian metric.

    Returns:
        (Tensor): squared riemannian distances
    """
    G = torch.matmul(se2_inverse(Gg), Gh)

    # transform group product to be sure the element is in the projective line bundle of the se2 group
    x, y, theta = se2_element(G)
    G = se2_matrix(x, y, theta)

    return weighted_norm(se2_log(G), Re)
