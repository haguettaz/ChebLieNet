import math

import torch

def metric_tensor(theta, l1, l2, l3):
    """
    [summary]

    Args:
        theta ([type]): [description]
        l1 ([type]): [description]
        l2 ([type]): [description]
        l3 ([type]): [description]

    Returns:
        [type]: [description]
    """
    e1 = torch.tensor([math.cos(theta), math.sin(theta), 0], dtype=torch.float32)
    e2 = torch.tensor([-math.sin(theta), math.cos(theta), 0], dtype=torch.float32)
    e3 = torch.tensor([0, 0, 1], dtype=torch.float32)

    D = e1.unsqueeze(1) * e1.unsqueeze(0) * l1
    D += e2.unsqueeze(1) * e2.unsqueeze(0) * l2
    D += e3.unsqueeze(1) * e3.unsqueeze(0) * l3

    return D