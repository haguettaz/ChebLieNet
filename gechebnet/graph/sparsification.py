import math
from typing import Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor

from ..utils import shuffle_tensor
from .signal_processing import get_laplacian
from .utils import code_edges


def get_sparse_laplacian(laplacian, sparsification_rate, on="edges"):
    if not on in {"edges", "nodes"}:
        raise ValueError(f"{on} is not a valid value for on: must be 'edges' or 'nodes'.")

    num_nodes = laplacian.size(0)

    # number of edges corresponds to non zero values minus the number of diagonal elements
    num_edges = laplacian._nnz() - num_nodes

    edge_index = laplacian._indices()
    edge_weight = -laplacian._values()
    mask = edge_index[0] != edge_index[1]  # mask corresponding to non-diagonal elements

    if on == "edges":
        return get_edges_sparse_laplacian(
            edge_index=edge_index[:, mask],
            edge_weight=edge_weight[mask],
            num_nodes=num_nodes,
            num_edges=num_edges,
            sparsification_rate=sparsification_rate,
        )

    return get_nodes_sparse_laplacian(
        edge_index=edge_index[:, mask],
        edge_weight=edge_weight[mask],
        node_index=edge_index[0].unique(),
        num_nodes=num_nodes,
        num_edges=num_edges,
        sparsification_rate=sparsification_rate,
    )


def get_edges_sparse_laplacian(edge_index, edge_weight, num_nodes, num_edges, sparsification_rate):
    edge_code = code_edges(edge_index, edge_weight, num_nodes)
    unique = edge_code.unique(return_inverse=True)

    num_samples = math.ceil((1 - sparsification_rate) * unique.size(0))  # num edges to keep
    probabilities = (
        unique - unique.floor()
    )  # weight corresponds to the decimal part of the edge coding
    random_sampling = torch.multinomial(probabilities, num_samples)

    mask = torch.tensor([False] * num_edges)
    for eidx in unique[random_sampling]:
        mask += edge_code == eidx

    return get_laplacian(edge_index[:, mask], edge_weight[mask], num_nodes)


def get_nodes_sparse_laplacian(
    edge_index, edge_weight, node_index, num_nodes, num_edges, sparsification_rate
):
    num_samples = math.floor(sparsification_rate * num_nodes)  # num nodes to drop
    random_sampling = torch.multinomial(torch.ones(num_nodes), num_samples)

    mask = torch.tensor([False] * num_edges)
    for nidx in node_index[random_sampling]:
        mask += (edge_index[0] == nidx) + (edge_index[1] == nidx)

    return get_laplacian(edge_index[:, ~mask], edge_weight[~mask], num_nodes)
