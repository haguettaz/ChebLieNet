from typing import Tuple

import torch
from torch import FloatTensor, LongTensor

from .compression import edge_compression


def remove_self_loops(
    edge_index: LongTensor, edge_weight: FloatTensor
) -> Tuple[LongTensor, FloatTensor]:
    """
    Removes every self-loop in the graph given by edge_index and edge_weight.

    Args:
        edge_index (LongTensor): indices of edges.
        edge_weight (FloatTensor): weights of edges.

    Returns:
        LongTensor: indices of edges.
        FloatTensor: weights of edges.
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask], edge_weight[mask]


def to_undirected(
    edge_index: LongTensor, edge_weight: FloatTensor
) -> Tuple[LongTensor, FloatTensor]:
    """
    Remove edges that are not symmetric.

    Args:
        edge_index (LongTensor): indices of edges.
        edge_weight (FloatTensor): weights of edges.

    Returns:
        LongTensor: indices of edges.
        FloatTensor: weights of edges.
    """

    edge_index_t = edge_index.t()
    max_, _ = torch.max(edge_index_t, dim=1)
    min_, _ = torch.min(edge_index_t, dim=1)
    minmax = torch.stack((min_, max_), dim=0)

    _, inverse, counts = torch.unique(minmax, dim=1, return_inverse=True, return_counts=True)
    bad_index = torch.arange(counts.shape[0])[(counts < 2)]

    mask = torch.tensor([False] * inverse.shape[0])
    for idx in bad_index:
        mask |= inverse == idx

    return edge_index[:, ~mask], edge_weight[~mask]


def process_edges(
    edge_index: LongTensor, edge_attr: FloatTensor, kappa: float
) -> Tuple[LongTensor, FloatTensor]:
    """
    Process edges of graph:
        1. Remove self-loops
        2. Remove directed edges
        3. Compress edges (optional)

    Args:
        edge_index (LongTensor): indices of edges.
        edge_attr (FloatTensor): attributes of edges.
        kappa (float): compression rate.

    Returns:
        (LongTensor): indices of edges.
        (FloatTensor): attributes of edges.
    """
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    if kappa > 0.0:
        edge_index, edge_attr = edge_compression(edge_index, edge_attr, kappa)
        print("Compression: Done!")
    return edge_index, edge_attr
