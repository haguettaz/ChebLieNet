# coding=utf-8

import torch
from scipy.linalg import eigh

from ..utils.utils import sparse_tensor_to_sparse_array
from .utils import add_self_loops, remove_self_loops


def get_laplacian(
    edge_index,
    edge_weight,
    num_vertices,
    device=None,
):
    """
    Return the symmetric normalized laplacian of the graph. Eigenvalues are in the range [0, 2].

    Args:
        edge_index (`torch.LongTensor`): edges' indices.
        edge_weight (`torch.FloatTensor`): edges' weights.
        num_vertices (int): number of vertices.
        device (`torch.device`, optional): computation device. Defaults to None.

    Returns:
        (`torch.sparse.FloatTensor`): symmetric normalized laplacian.
    """

    node_degree = torch.zeros(num_vertices).scatter_add(0, edge_index[0], edge_weight)

    node_degree_norm = node_degree.pow(-0.5)
    edge_weight = -(edge_weight * node_degree_norm[edge_index[0]] * node_degree_norm[edge_index[1]])

    edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    laplacian = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size((num_vertices, num_vertices)))
    laplacian = laplacian.to(device)
    return laplacian


def get_rescaled_laplacian(
    edge_index,
    edge_weight,
    num_vertices,
    lmax=2.0,
    device=None,
):
    """
    Get rescaled symmetric normalized laplacian. Eigenvalues are in the range [-1, 1].

    Args:
        edge_index (`torch.LongTensor`): edges' indices.
        edge_weight (`torch.FloatTensor`): edges' weights.
        num_vertices (int): number of vertices.
        lmax (float): maximum eigenvalue. Defaults to 2.0.
        device (`torch.device`, optional): computation device. Defaults to None.

    Returns:
        (`torch.sparse.FloatTensor`): rescaled symmetric normalized laplacian.
    """
    node_degree = torch.zeros(num_vertices).scatter_add(0, edge_index[0], edge_weight)

    node_degree_norm = node_degree.pow(-0.5)
    edge_weight = -2 / lmax * (edge_weight * node_degree_norm[edge_index[0]] * node_degree_norm[edge_index[1]])

    laplacian = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size((num_vertices, num_vertices)))
    laplacian = laplacian.to(device)
    return laplacian


def get_fourier_basis(laplacian):
    """
    Return graph Fourier basis, i.e. the eigen decomposition of the graph's laplacian.

    Args:
        laplacian (`torch.sparse.FloatTensor`): graph's laplacian.

    Returns:
        (`torch.FloatTensor`): eigenvalues of the graph's laplacian.
        (`torch.FloatTensor`): eigenvectors of the graph's laplacian
    """
    lambdas, Phi = eigh(sparse_tensor_to_sparse_array(laplacian).toarray(), driver="ev")

    return torch.from_numpy(lambdas), torch.from_numpy(Phi)
