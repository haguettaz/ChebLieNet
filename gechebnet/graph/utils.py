from typing import Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor
from torch import device as Device



def code_edges(edge_index: LongTensor, edge_weight: FloatTensor, num_nodes: int) -> FloatTensor:
    """
    Generates a coded tensor corresponding to edges' indices and weights. Let s(e) (resp. t(e)) be the indices
    of the source (resp. target) of the edge e with weight w(e) and let N be the total number of nodes. The code
    is as follows:
        c(e) = min (s(e), t(e)) + N * |s(e) - t(e)| + w(e)

    Args:
        edge_index (LongTensor): indices of edges.
        edge_weight (FloatTensor): weights of edges.
        num_nodes (int): number of nodes.

    Raises:
        ValueError: maximum weights must be strictly lower than 1, otherwise, the code is not uniquely decodable
        (up to edge sens).

    Returns:
        FloatTensor: coded edges.
    """
    if edge_weight.max() >= 1:
        raise ValueError(
            f"Found at least one weight higher or equal to 1, the edge's code cannot work"
        )

    val_min, _ = edge_index.min(dim=0)
    val_diff = (edge_index[1] - edge_index[0]).abs()

    return val_min * num_nodes + val_diff + edge_weight


def remove_duplicated_edges(
    edge_index: LongTensor, edge_attr: FloatTensor, knn: int
) -> Tuple[LongTensor, FloatTensor]:
    """
    Remove duplicated edges in the graph given by edge_index and edge_attr. Duplicated edges can appear if the
    number of connections per vertex is higher than the total number of vertices.

    Args:
        edge_index (LongTensor): indices of edges.
        edge_attr (FloatTensor): attributes of edges.
        knn (int): number of connections of a vertex.

    Raises:
        ValueError: number of edges must be equal to the number of nodes times the number of connections per node.

    Returns:
        (LongTensor): indices of edges.
        (FloatTensor)]: attributes of edges
    """
    num_nodes, num_edges = edge_index.max() + 1, edge_index.shape[1]

    if knn * num_nodes != num_edges:
        raise ValueError(
            f"The number of edges {num_edges} does not coincide with the number of nodes {num_nodes} and knn {knn}"
        )

    indices = [idx for idx in range(num_edges) if idx % knn < num_nodes]

    return edge_index[:, indices], edge_attr[indices]


def remove_self_loops(
    edge_index: LongTensor, edge_attr: FloatTensor
) -> Tuple[LongTensor, FloatTensor]:
    """
    Removes every self-loop in the graph given by edge_index and edge_attr.

    Args:
        edge_index (LongTensor): indices of edges.
        edge_attr (FloatTensor): attributes of edges.

    Returns:
        LongTensor: indices of edges.
        FloatTensor: attributes of edges.
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask], edge_attr[mask]


def remove_directed_edges(
    edge_index: LongTensor, edge_weight: FloatTensor, num_nodes: int
) -> Tuple[LongTensor, FloatTensor]:
    """
    Removes every directed edges in the graph given by edge_index and edge_weight.

    Args:
        edge_index (LongTensor): indices of edges.
        edge_weight (FloatTensor): weights of edges.
        num_nodes (int): number of nodes.

    Returns:
        LongTensor: indices of edges.
        FloatTensor: weights of edges.
    """

    # codes the edges to use unique function of pytorch
    coded_edges = code_edges(edge_index, edge_weight, num_nodes)

    num_edges = edge_index.shape[1]

    _, indices, counts = torch.unique(coded_edges, return_inverse=True, return_counts=True)
    counts_indices = torch.arange(counts.shape[0])

    mask = BoolTensor([False] * num_edges)
    for c_i in counts_indices[counts < 2]:  # directed edges
        mask |= indices == c_i

    return edge_index[:, ~mask], edge_weight[~mask]


