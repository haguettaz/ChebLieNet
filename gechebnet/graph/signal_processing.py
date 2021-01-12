from typing import Tuple

import torch
from numpy import ndarray
from numpy.linalg import eigh
from torch import FloatTensor, LongTensor
from torch.sparse import FloatTensor as SparseFloatTensor

from ..utils import sparse_tensor_diag, sparse_tensor_to_sparse_array


def get_laplacian(
    edge_index: LongTensor, edge_weight: FloatTensor, num_nodes: int
) -> SparseFloatTensor:
    """
    Get symmetric normalized laplacian from edge indices and weights.

    Args:
        edge_index (LongTensor): edge indices.
        edge_weight (FloatTensor): edge weights.
        num_nodes (int): number of nodes.

    Returns:
        SparseFloatTensor: symmetric normalized laplacian.
    """

    deg = torch.zeros(num_nodes).scatter_add(0, edge_index[1], edge_weight).pow(-0.5)
    edge_weight = edge_weight * deg[edge_index[0]] * deg[edge_index[1]]
    W_norm = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size((num_nodes, num_nodes)))
    I = sparse_tensor_diag(num_nodes)
    return I - W_norm


def get_fourier_basis(laplacian: SparseFloatTensor, tol=1e-2) -> Tuple[ndarray, ndarray]:
    """
    Return graph Fourier basis, i.e. Laplacian eigen decomposition.

    Args:
        laplacian (SparseFloatTensor): graph laplacian.
        tol (float): tolerance for negative eigen values. Defaults to 1e-2.

    Returns:
        (ndarray): Laplacian eigen values.
        (ndarray): Laplacian eigen vectors.
    """
    lambdas, Phi = eigh(sparse_tensor_to_sparse_array(laplacian).toarray())
    if lambdas.min() < -tol:
        raise ValueError(
            f"An error occured in the eigen decomposition of the graph laplacian. The eigen values must be positive"
        )

    else:
        lambdas[lambdas < 0] = 0.0
    return lambdas, Phi
