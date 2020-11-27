import math

import torch

from .utils.utils import shuffle

def metric_tensor(theta, l1, l2, l3):
    e1 = torch.tensor([math.cos(theta), math.sin(theta), 0], dtype=torch.float32)
    e2 = torch.tensor([-math.sin(theta), math.cos(theta), 0], dtype=torch.float32)
    e3 = torch.tensor([0, 0, 1], dtype=torch.float32)

    D = e1.unsqueeze(1) * e1.unsqueeze(0) * l1
    D += e2.unsqueeze(1) * e2.unsqueeze(0) * l2
    D += e3.unsqueeze(1) * e3.unsqueeze(0) * l3

    return D


def static_node_compression(node_index, kappa):
    num_nodes = node_index.nelement()

    num_to_remove = int(kappa * num_nodes)
    num_to_keep = num_nodes - num_to_remove

    mask = torch.tensor([False] * num_to_remove + [True] * num_to_keep)
    mask = shuffle(mask)

    return node_index[mask]
