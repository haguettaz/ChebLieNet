from typing import Optional, Tuple

import torch
from numpy import ndarray
from scipy.linalg import eigh
from torch import FloatTensor, LongTensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor

from ..utils import sparse_tensor_to_sparse_array
from .utils import add_self_loops, remove_self_loops


def get_laplacian(
    edge_index: LongTensor,
    edge_weight: FloatTensor,
    num_nodes: int,
    device: Optional[Device] = None,
) -> SparseFloatTensor:
    """
    Get symmetric normalized laplacian from edge indices and weights. Eigenvalues are in the range [0, 2].

    Args:
        edge_index (LongTensor): edge indices.
        edge_weight (FloatTensor): edge weights.
        num_nodes (int): number of nodes.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        SparseFloatTensor: symmetric normalized laplacian.
    """

    node_degree = torch.zeros(num_nodes).scatter_add(0, edge_index[0], edge_weight)

    inv_sqrt_node_degree = node_degree.pow(-0.5)
    edge_weight = -(edge_weight * inv_sqrt_node_degree[edge_index[0]] * inv_sqrt_node_degree[edge_index[1]])

    edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    laplacian = SparseFloatTensor(edge_index, edge_weight, torch.Size((num_nodes, num_nodes)))
    laplacian = laplacian.to(device)
    return laplacian


def get_norm_laplacian(
    edge_index: LongTensor,
    edge_weight: FloatTensor,
    num_nodes: int,
    lmax: float = 2.0,
    device: Optional[Device] = None,
) -> SparseFloatTensor:
    """
    Get rescaled symmetric normalized laplacian from edge indices and weights. Eigenvalues are normalized in the
    range [-1, 1].

    Args:
        edge_index (LongTensor): edge indices.
        edge_weight (FloatTensor): edge weights.
        num_nodes (int): number of nodes.
        lmax (float): highest eigenvalue. Defaults to 2.0.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        SparseFloatTensor: symmetric normalized laplacian with rescaled eigenvalues.
    """
    node_degree = torch.zeros(num_nodes).scatter_add(0, edge_index[0], edge_weight)

    node_degree_norm = node_degree.pow(-0.5)
    edge_weight = -2 / lmax * (edge_weight * node_degree_norm[edge_index[0]] * node_degree_norm[edge_index[1]])

    laplacian = SparseFloatTensor(edge_index, edge_weight, torch.Size((num_nodes, num_nodes)))
    laplacian = laplacian.to(device)
    return laplacian


def get_fourier_basis(laplacian: SparseFloatTensor) -> Tuple[ndarray, ndarray]:
    """
    Return graph Fourier basis, i.e. Laplacian eigen decomposition.

    Args:
        laplacian (SparseFloatTensor): graph laplacian.

    Returns:
        (ndarray): Laplacian eigenvalues.
        (ndarray): Laplacian eigenvectors.
    """
    lambdas, Phi = eigh(sparse_tensor_to_sparse_array(laplacian).toarray(), driver="ev")

    return lambdas, Phi
