import math
from typing import Optional, Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor

from ..utils import shuffle_tensor
from .signal_processing import get_laplacian
from .utils import remove_duplicated_edges, remove_self_loops, to_undirected


def sparsify_on_edges(
    edge_index: LongTensor,
    edge_weight: FloatTensor,
    rate: float,
) -> Tuple[LongTensor, FloatTensor]:
    """
    [summary]

    Args:
        edge_index (LongTensor): [description]
        edge_weight (FloatTensor): [description]
        num_nodes (int): [description]
        sparsification_rate (float): [description]

    Returns:
        SparseFloatTensor: [description]
    """

    edge_index, edge_weight = remove_duplicated_edges(edge_index, edge_weight, self_loop=False)

    num_samples = math.ceil((1 - rate) * edge_weight.size(0))  # num edges to keep

    random_sampling = torch.multinomial(edge_weight, num_samples)

    edge_index, edge_weight = to_undirected(edge_index[:, random_sampling], edge_weight[random_sampling])

    return edge_index, edge_weight


def sparsify_on_nodes(
    edge_index: LongTensor,
    edge_weight: FloatTensor,
    node_index: LongTensor,
    num_nodes: int,
    rate: float,
) -> Tuple[LongTensor, FloatTensor]:
    """
    [summary]

    Args:
        edge_index (LongTensor): [description]
        edge_weight (FloatTensor): [description]
        num_nodes (int): [description]
        sparsification_rate (float): [description]

    Returns:
        SparseFloatTensor: [description]
    """

    num_samples = math.floor(rate * num_nodes)
    if num_samples < 1:
        return edge_index, edge_weight

    random_sampling = torch.multinomial(torch.ones(num_nodes), num_samples)

    mask = torch.ones(edge_weight.shape, dtype=torch.bool)
    for nidx in node_index[random_sampling]:
        mask &= (edge_index[0] != nidx) & (edge_index[1] != nidx)

    return edge_index[:, mask], edge_weight[mask]
