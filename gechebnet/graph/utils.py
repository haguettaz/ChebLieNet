import math

import torch


def metric_tensor(theta, l1, l2, l3):
    """
    Return the anisotropic metric tensor, the main directions of the kernel are:
        1. aligned with theta and orthogonal to the orientation axis.
        2. orthogonal to theta and to the orientation axis.
        3. aligned with the orientation axis.

    Args:
        theta (float): the orientation of the first main direction of the kernel (in radians).
        l1 (float): the intensity of the first main direction.
        l2 (float): the intensity of the second main direction.
        l3 (float): the intensity of the third main direction.

    Returns:
        (torch.tensor): the metric tensor with shape (3, 3).
    """
    e1 = torch.tensor([math.cos(theta), math.sin(theta), 0], dtype=torch.float32)
    e2 = torch.tensor([-math.sin(theta), math.cos(theta), 0], dtype=torch.float32)
    e3 = torch.tensor([0, 0, 1], dtype=torch.float32)

    D = e1.unsqueeze(1) * e1.unsqueeze(0) * l1
    D += e2.unsqueeze(1) * e2.unsqueeze(0) * l2
    D += e3.unsqueeze(1) * e3.unsqueeze(0) * l3

    return D
