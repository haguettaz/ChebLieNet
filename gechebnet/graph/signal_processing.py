import torch
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from torch_scatter import scatter_add

from ..utils import sparse_tensor_diag, sparse_tensor_to_sparse_array
from .utils import is_undirected


def get_laplacian(edge_index, edge_weight, num_nodes, norm=None):

    deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=num_nodes)

    if norm is None:
        D = sparse_tensor_diag(num_nodes, deg)
        W = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size((num_nodes, num_nodes)))
        return D - W

    if not norm in ["sym", "rw"]:
        raise ValueError(f"{norm} is an invalid value for parameter norm")

    if norm == "sym":
        deg_sqrt_inv = deg.pow(-0.5)
        edge_weight = edge_weight * deg_sqrt_inv[edge_index[0]] * deg_sqrt_inv[edge_index[1]]
        W_norm = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size((num_nodes, num_nodes)))
        I = sparse_tensor_diag(num_nodes)
        return I - W_norm

    if norm == "rw":
        deg_inv = deg.pow(-1)
        edge_weight = edge_weight * deg_inv[edge_index[0]]
        W_norm = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size((num_nodes, num_nodes)))
        I = sparse_tensor_diag(num_nodes)
        return I - W_norm


def get_fourier_basis(edge_index, edge_weight, num_nodes, norm=None):
    if not is_undirected(edge_index, edge_weight):
        raise ValueError("The graph is directed, the laplacian might do not have a proper eigen decomposition")

    laplacian = get_laplacian(edge_index, edge_weight, num_nodes, norm)
    lambdas, Phi = eigh(sparse_tensor_to_sparse_array(laplacian).toarray())
    return lambdas, Phi
