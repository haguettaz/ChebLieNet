import math
from typing import Optional, Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor

from ..utils import shuffle_tensor
from .signal_processing import get_laplacian
from .utils import code_edges


def get_sparse_laplacian(
    laplacian: SparseFloatTensor,
    on: Optional[str] = "edges",
    rate: Optional[float] = 0.3,
    device: Optional[Device] = None,
) -> SparseFloatTensor:
    """
    [summary]

    Args:
        laplacian (SparseFloatTensor): [description]
        sparsification_rate (float): [description]
        on (Optional[str], optional): [description]. Defaults to "edges".

    Raises:
        ValueError: [description]

    Returns:
        SparseFloatTensor: [description]
    """
    if not on in {"edges", "nodes"}:
        raise ValueError(f"{on} is not a valid value for on: must be 'edges' or 'nodes'.")

    if rate == 0:
        return laplacian

    edge_index = laplacian._indices()
    edge_weight = -laplacian._values()

    # since the graph is without self-loop, the edges are contained on the non diagonal elements
    mask = edge_index[0] != edge_index[1]

    if on == "edges":
        return get_edges_sparse_laplacian(
            edge_index=edge_index[:, mask],
            edge_weight=edge_weight[mask],
            num_nodes=laplacian.size(0),
            rate=rate,
            device=device,
        )

    return get_nodes_sparse_laplacian(
        edge_index=edge_index[:, mask],
        edge_weight=edge_weight[mask],
        num_nodes=laplacian.size(0),
        rate=rate,
        device=device,
    )


def get_edges_sparse_laplacian(
    edge_index: LongTensor,
    edge_weight: FloatTensor,
    num_nodes: int,
    rate: float,
    device: Device,
) -> SparseFloatTensor:
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
    edge_code = code_edges(edge_index, edge_weight, num_nodes)
    unique = edge_code.unique()

    num_samples = math.ceil((1 - rate) * unique.size(0))  # num edges to keep

    # weight corresponds to the decimal part of the edge coding
    probabilities = unique - unique.floor()
    random_sampling = torch.multinomial(probabilities, num_samples)

    mask = torch.zeros(edge_code.shape, dtype=torch.bool)

    for eidx in unique[random_sampling]:
        mask |= edge_code == eidx

    return get_laplacian(edge_index[:, mask], edge_weight[mask], num_nodes, device)


def get_nodes_sparse_laplacian(
    edge_index: LongTensor,
    edge_weight: FloatTensor,
    num_nodes: int,
    rate: float,
    device: Device,
) -> SparseFloatTensor:
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
    node_index = edge_index[0].unique()

    num_samples = math.floor(rate * num_nodes)  # num nodes to drop
    random_sampling = torch.multinomial(torch.ones(num_nodes), num_samples)

    mask = torch.zeros(edge_weight.shape, dtype=torch.bool)
    for nidx in node_index[random_sampling]:
        mask |= (edge_index[0] == nidx) | (edge_index[1] == nidx)

    return get_laplacian(edge_index[:, ~mask], edge_weight[~mask], num_nodes, device)
