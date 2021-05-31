import math

import torch


def rotation_matrix(angle, axis, device=None):
    """
    Return a new tensor filled with rotation matrices.

    Args:
        angle (`torch.FloatTensor`): rotation angles.
        axis (str): rotation axis, e.g. 'x' or 'y' or 'z'.
        device (`torche.device`, optional): computation device. Defaults to None.

    Returns:
        (`torch.FloatTensor`): rotation matrices.
    """
    R = torch.zeros(angle.nelement(), 3, 3, device=device)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    if axis == "x":
        R[..., 0, 0] = 1.0
        R[..., 1, 1] = cos
        R[..., 1, 2] = -sin
        R[..., 2, 1] = sin
        R[..., 2, 2] = cos

    if axis == "y":
        R[..., 0, 0] = cos
        R[..., 0, 2] = sin
        R[..., 2, 0] = -sin
        R[..., 2, 2] = cos
        R[..., 1, 1] = 1.0

    if axis == "z":
        R[..., 0, 0] = cos
        R[..., 0, 1] = -sin
        R[..., 1, 0] = sin
        R[..., 1, 1] = cos
        R[..., 2, 2] = 1.0

    return R


def xyz2betagamma(x, y, z):
    """
    Return new tensors corresponding to angle representation from the cartesian representation.

    Warning: x, y, z have to be on the 1-sphere.

    Args:
        x (`torch.FloatTensor`): x positions.
        y (`torch.FloatTensor`): y positions.
        z (`torch.FloatTensor`): z positions.

    Returns:
        (`torch.FloatTensor`): beta rotations about y axis.
        (`torch.FloatTensor`): gamma rotations about z axis.
    """
    beta = torch.acos(z)
    gamma = torch.atan2(y, x)
    return beta, gamma


def betagamma2xyz(beta, gamma, axis=None):
    """
    Returns new tensors corresponding to angle representation from the cartesian representation.

    Args:
        beta (`torch.FloatTensor`): beta rotations about y axis.
        gamma (`torch.FloatTensor`): gamma rotations about z axis.
        axis (str, optional): cartesian axis. If None, return all axis. Defaults to None.

    Returns:
        (`torch.FloatTensor`, optional): x positions.
        (`torch.FloatTensor`, optional): y positions.
        (`torch.FloatTensor`, optional): z positions.
    """

    if axis == "x":
        return torch.sin(beta) * torch.cos(gamma)
    if axis == "y":
        return torch.sin(beta) * torch.sin(gamma)
    if axis == "z":
        return torch.cos(beta)

    x = torch.sin(beta) * torch.cos(gamma)
    y = torch.sin(beta) * torch.sin(gamma)
    z = torch.cos(beta)

    return x, y, z
