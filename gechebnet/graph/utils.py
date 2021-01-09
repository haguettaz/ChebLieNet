import math
from typing import Optional, Tuple

import torch
from pykeops.torch import LazyTensor, Pm
from torch_sparse import coalesce, transpose

from ..utils import lower, upper
from .compression import edge_compression


def group_inverse(x):
    return LazyTensor.cat(
        (-x[0] * x[2].cos() - x[1] * x[2].sin(), -x[1] * x[2].cos() + x[0] * x[2].sin(), -x[2]), dim=-1
    )


def group_product(x1, x2):
    return LazyTensor.cat(
        (
            x1[0] + x2[0] * (x1[2]).cos() - x2[1] * (x1[2]).sin(),
            x1[1] + x2[1] * (x1[2]).cos() + x2[0] * (x1[2]).sin(),
            x1[2] + x2[2],
        ),
        dim=-1,
    )


def group_log(x):
    # if theta == 0, cot(theta) return 0, we use this implementation detail for the log expression by part
    eps = 1e-5
    return LazyTensor.cat(
        (
            lower(x[2].mod(math.pi, 0.0).abs(), eps) * x[0]
            + upper(x[2].mod(math.pi, 0.0).abs(), eps)
            * 0.5
            * x[2].mod(math.pi, -math.pi / 2)
            * (x[1] + x[0] * ((x[2].mod(math.pi, -math.pi / 2) / 2).cot())),
            lower(x[2].mod(math.pi, 0.0).abs(), eps) * x[1]
            + upper(x[2].mod(math.pi, 0.0).abs(), eps)
            * 0.5
            * x[2].mod(math.pi, -math.pi / 2)
            * (-x[0] + x[1] * ((x[2].mod(math.pi, -math.pi / 2) / 2).cot())),
            x[2].mod(math.pi, -math.pi / 2),
        ),
        dim=-1,
    )


def metric_tensor(sigmas, device):
    return Pm(torch.tensor([*sigmas], device=device))


def square_distance(xi: LazyTensor, xj: LazyTensor, S: LazyTensor) -> LazyTensor:
    r"""Returns the square distance based on the delta position in :attr:`dx` and the metric tensor
    in :attr:`S`:

    :math:`d(x,y) = (x-y)^\top \Sigma (x-y)`

    Args:
        dx (LazyTensor): delta position between source's and target's points with shape (N, M, 3)
        S (LazyTensor): metric tensor with shape (N, M, 3, 3).

    Returns:
        LazyTensor: square distances
    """

    v = group_log(group_product(group_inverse(xi), xj))

    return v.weightedsqnorm(S)


# def square_distance(xi, xj, S):
#     xi_inv = group_inverse(xi)
#     v = group_product(xi_inv, xj)
#     # v = pl_log(xi_inv_xj)
#     return v.keops_tensordot(v, (1, 3), (3, 1), (1,), (0,))

#     # v = pl_log(group_product(group_inverse(xi), xj))
#     # return v.keops_tensordot(S, (1, 3), (3, 3), (1,), (0,)).keops_tensordot(v, (3, 3), (3, 1), (1,), (0,))


# def metric_tensor(
#     abs_dx3: LazyTensor, sigmas: Tuple[float, float, float], device: Optional[torch.device] = None
# ) -> LazyTensor:
#     r"""Return the anisotropic metric tensor, based on the angles' differences given by :attr:`abs_dx3`. The main
#     directions are:
#         1. aligned with theta and orthogonal to the orientation axis.
#         2. orthogonal to theta and to the orientation axis.
#         3. aligned with the orientation axis.

#     The metric tensor is computed via its eigen decomposition :math:`S = U \Lambda U^\top`.

#     Args:
#         abs_dx3 (LazyTensor): angle's differences' with shape (N, M).
#         sigmas (Tuple[float, float, float]): intensities of the three main anisotropic directions.
#         device (Optional[torch.device]): device to use. Defaults to None.

#     Returns:
#         LazyTensor: metric tensor with shape (N, M, 3, 3).
#     """
#     device = device or torch.device("cpu")

#     s1, s2, s3 = sigmas

#     L = Pm(torch.tensor([s1, 0.0, 0.0, 0.0, s2, 0.0, 0.0, 0.0, s3], device=device))

#     U = LazyTensor.cat(
#         (abs_dx3.cos(), -(abs_dx3.sin()), Pm(0.0), abs_dx3.sin(), abs_dx3.cos(), Pm(0.0), Pm(0.0), Pm(0.0), Pm(1.0)),
#         dim=-1,
#     )

#     U_t = LazyTensor.cat(
#         (abs_dx3.cos(), abs_dx3.sin(), Pm(0.0), -(abs_dx3.sin()), abs_dx3.cos(), Pm(0.0), Pm(0.0), Pm(0.0), Pm(1.0)),
#         dim=-1,
#     )

#     S = U.keops_tensordot(L, (3, 3), (3, 3), (1,), (0,)).keops_tensordot(U_t, (3, 3), (3, 3), (1,), (0,))

#     return S


def delta_pos(xi: LazyTensor, xj: LazyTensor) -> LazyTensor:
    r"""Return the delta position along each dimension. The position's format is:
        - x1 : first spatial dimension
        - x2 : second spatial dimension
        - x3 : orientation angle dimension, it is pi-periodic

    Args:
        xi (LazyTensor): source points' positions with shape (N, 1, 3)
        xj (LazyTensor): target's points' positions (1, M, 3)

    Returns:
        LazyTensor: pairwise delta position on each dimension between :attr:`xi` and :attr:`xj`
    """
    dx1 = xj[0] - xi[0]
    dx2 = xj[1] - xi[1]
    dx3 = mod(xj[2] - xi[2], math.pi, -math.pi / 2)

    dx = LazyTensor.cat((dx1, dx2, dx3), dim=-1)

    return dx


def remove_self_loops(edge_index: torch.LongTensor, edge_weight: torch.Tensor):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask], edge_weight[mask]


def is_undirected(edge_index, edge_weight, num_nodes):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` is
    undirected.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool
    """

    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)

    edge_index_t, edge_weight_t = transpose(edge_index, edge_weight, num_nodes, num_nodes, coalesced=True)
    index_symmetric = torch.all(edge_index == edge_index_t)
    weight_symmetric = torch.all(edge_weight == edge_weight_t)
    return index_symmetric and weight_symmetric


def to_undirected(edge_index, edge_weight):
    """
    Remove edges that are not symmetric

    Args:
        edge_index ([type]): [description]
        edge_weight ([type]): [description]

    Returns:
        [type]: [description]
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


def process_edges(edge_index: torch.tensor, edge_sqdist: torch.tensor, kappa: float):
    """
    [summary]

    Args:
        edge_index (torch.tensor): [description]
        edge_sqdist (torch.tensor): [description]
        kappa (float): [description]

    Returns:
        [type]: [description]
    """
    edge_index, edge_sqdist = remove_self_loops(edge_index, edge_sqdist)
    edge_index, edge_sqdist = to_undirected(edge_index, edge_sqdist)
    if kappa > 0.0:
        edge_index, edge_sqdist = edge_compression(edge_index, edge_sqdist, kappa)
    return edge_index, edge_sqdist
