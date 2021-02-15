from typing import Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor


def remove_duplicated_edges(
    edge_index: LongTensor, edge_attr: FloatTensor, self_loop=False
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

    if self_loop:
        mask = edge_index[0] <= edge_index[1]
    else:
        mask = edge_index[0] < edge_index[1]

    return edge_index[:, mask], edge_attr[..., mask]


def to_undirected(edge_index: LongTensor, edge_attr: FloatTensor) -> Tuple[LongTensor, FloatTensor]:
    """
    [summary]

    Args:
        edge_index (LongTensor): [description]
        edge_attr (FloatTensor): [description]

    Returns:
        (LongTensor): [description]
        (FloatTensor): [description]
    """
    edge_index_inverse = torch.cat((edge_index[1, None], edge_index[0, None]), dim=0)
    edge_index = torch.cat((edge_index, edge_index_inverse), dim=-1)
    edge_attr = torch.cat((edge_attr, edge_attr), dim=-1)
    return edge_index, edge_attr


def remove_self_loops(edge_index: LongTensor, edge_attr: FloatTensor) -> Tuple[LongTensor, FloatTensor]:
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


def add_self_loops(
    edge_index: LongTensor, edge_attr: FloatTensor, weight: float = 1.0
) -> Tuple[LongTensor, FloatTensor]:
    """
    [summary]

    Args:
        edge_index (LongTensor): [description]
        edge_attr (FloatTensor): [description]
        weight (float, optional): [description]. Defaults to 1.0.

    Returns:
        Tuple[LongTensor, FloatTensor]: [description]
    """

    self_loop_index = edge_index[0].unique().unsqueeze(0).repeat(2, 1)
    self_loop_attr = weight * torch.ones(self_loop_index.shape[1])

    edge_index = torch.cat((self_loop_index, edge_index), dim=1)
    edge_attr = torch.cat((self_loop_attr, edge_attr))

    return edge_index, edge_attr
