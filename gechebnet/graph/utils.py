from typing import Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor

from .compression import compress_edges


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


# def to_undirected(
#     edge_index: LongTensor, edge_weight: FloatTensor
# ) -> Tuple[LongTensor, FloatTensor]:
#     """
#     Remove edges that are not symmetric.

#     Args:
#         edge_index (LongTensor): indices of edges.
#         edge_weight (FloatTensor): weights of edges.

#     Returns:
#         LongTensor: indices of edges.
#         FloatTensor: weights of edges.
#     """

#     edge_index_t = edge_index.t()
#     max_, _ = torch.max(edge_index_t, dim=1)
#     min_, _ = torch.min(edge_index_t, dim=1)
#     minmax = torch.stack((min_, max_), dim=0)

#     _, inverse, counts = torch.unique(minmax, dim=1, return_inverse=True, return_counts=True)
#     bad_index = torch.arange(counts.shape[0])[(counts < 2)]

#     mask = torch.tensor([False] * inverse.shape[0])
#     for idx in bad_index:
#         mask |= inverse == idx

#     return edge_index[:, ~mask], edge_weight[~mask]


def process_edges(
    edge_index: LongTensor, edge_weight: FloatTensor, knn: int, kappa: float
) -> Tuple[LongTensor, FloatTensor]:
    """
    Process edges of graph:
        1. Remove self-loops
        2. Remove directed edges
        3. Compress edges (optional)

    Args:
        edge_index (LongTensor): indices of edges.
        edge_weight (FloatTensor): weights of edges.
        kappa (float): compression rate.

    Returns:
        (LongTensor): indices of edges.
        (FloatTensor): weights of edges.
    """

    # remove duplicated edges due to too high knn
    edge_index, edge_weight = remove_duplicated_edges(edge_index, edge_weight, knn)

    # remove self loops
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    # codes the edges to use unique function of pytorch
    coded_edges = code_edges(edge_index, edge_weight)

    # remove directed edges
    edge_index, edge_weight = remove_directed_edges(edge_index, edge_weight, coded_edges)

    # compress graph
    if kappa > 0.0:
        edge_index, edge_weight = compress_edges(edge_index, edge_weight, kappa)

    return edge_index, edge_weight


def code_edges(edge_index, edge_weight):

    num_nodes = edge_index.max() + 1

    if edge_weight.max() >= 1:
        raise ValueError(
            f"Found at least one weight higher or equal to 1, the edge's code cannot work"
        )

    val_min, _ = edge_index.min(dim=0)
    val_diff = (edge_index[1] - edge_index[0]).abs()

    return val_min + num_nodes * val_diff + edge_weight


def remove_duplicated_edges(edge_index, edge_weight, knn):
    num_nodes, num_edges = edge_index.max() + 1, edge_index.shape[1]

    if knn * num_nodes != num_edges:
        raise ValueError(
            f"The number of edges {num_edges} does not coincide with the number of nodes {num_nodes} and knn {knn}"
        )

    indices = [idx for idx in range(num_edges) if idx % knn <= num_nodes]

    return edge_index[:, indices], edge_weight[indices]


def remove_directed_edges(edge_index, edge_weight, coded_edges):
    num_edges = edge_index.shape[1]

    _, indices, counts = torch.unique(coded_edges, return_inverse=True, return_counts=True)
    counts_indices = torch.arange(counts.shape[0])

    mask = BoolTensor([False] * num_edges)
    for c_i in counts_indices[counts == 1]:
        mask |= indices == c_i

    return edge_index[:, ~mask], edge_weight[~mask]


# def remove_self_loops(edge_index, edge_weight, coded_edges):
#     num_nodes = edge_index.max() + 1

#     mask = coded_edges < num_nodes

#     return edge_index[:, mask], edge_weight[mask]
