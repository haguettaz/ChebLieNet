import math
from typing import Tuple

import torch
from torch import FloatTensor

from ..utils import mod


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
