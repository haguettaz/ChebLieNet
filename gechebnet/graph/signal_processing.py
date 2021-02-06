from typing import Optional, Tuple

import torch
from numpy import ndarray
from scipy.linalg import eigh
from torch import FloatTensor, LongTensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor

from ..utils import sparse_tensor_diag, sparse_tensor_to_sparse_array
from .utils import remove_self_loops


def get_laplacian(
    edge_index: LongTensor,
    edge_weight: FloatTensor,
    num_nodes: int,
    device: Optional[Device] = None,
) -> SparseFloatTensor:
    """
    Get symmetric normalized laplacian from edge indices and weights.

    Args:
        edge_index (LongTensor): edge indices.
        edge_weight (FloatTensor): edge weights.
        num_nodes (int): number of nodes.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        SparseFloatTensor: symmetric normalized laplacian.
    """

    node_degree = torch.zeros(num_nodes).scatter_add(0, edge_index[1], edge_weight)
    inv_sqrt_node_degree = node_degree.pow(-0.5)
    edge_weight = -(
        edge_weight * inv_sqrt_node_degree[edge_index[0]] * inv_sqrt_node_degree[edge_index[1]]
    )

    diag_index = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
    diag_weight = torch.ones(num_nodes)
    mask = node_degree > 0

    index = torch.cat((diag_index[:, mask], edge_index), dim=1)
    weight = torch.cat((diag_weight[mask], edge_weight))

    return SparseFloatTensor(index, weight, torch.Size((num_nodes, num_nodes))).to(device)


def get_norm_laplacian(
    laplacian: SparseFloatTensor, device: Optional[Device] = None
) -> SparseFloatTensor:
    indices, values = remove_self_loops(laplacian._indices(), laplacian._values())
    return SparseFloatTensor(indices, values, laplacian.size()).to(device)


def get_fourier_basis(laplacian: SparseFloatTensor) -> Tuple[ndarray, ndarray]:
    """
    Return graph Fourier basis, i.e. Laplacian eigen decomposition.

    Args:
        laplacian (SparseFloatTensor): graph laplacian.

    Returns:
        (ndarray): Laplacian eigen values.
        (ndarray): Laplacian eigen vectors.
    """
    lambdas, Phi = eigh(sparse_tensor_to_sparse_array(laplacian).toarray(), driver="ev")

    return lambdas, Phi
