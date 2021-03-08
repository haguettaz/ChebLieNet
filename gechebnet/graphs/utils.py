# coding=utf-8

import torch


def remove_duplicated_edges(edge_index, edge_attr):
    """
    Remove duplicated edges in the graph, assuming an undirected graph.

    Args:
        edge_index (`torch.LongTensor`): indices of graph's edges.
        edge_attr (`torch.FloatTensor`): attributes of graph's edges.

    Returns:
        (`torch.LongTensor`): indices of graph's edges.
        (`torch.FloatTensor`): attributes of graph's edges.
    """
    mask = edge_index[0] <= edge_index[1]

    return edge_index[:, mask], edge_attr[..., mask]


def to_undirected(edge_index, edge_attr, self_loop=False):
    """
    Make the graph undirected, that is create an inverse edge for each edge.

    Args:
        edge_index (`torch.LongTensor`): indices of graph's edges.
        edge_attr (`torch.FloatTensor`): attributes of graph's edges.

    Returns:
        (`torch.LongTensor`): indices of graph's edges.
        (`torch.FloatTensor`): attributes of graph's edges.
    """
    edge_index_inverse = torch.cat((edge_index[1, None], edge_index[0, None]), dim=0)
    edge_index = torch.cat((edge_index, edge_index_inverse), dim=-1)
    edge_attr = torch.cat((edge_attr, edge_attr), dim=-1)

    if self_loop:
        return edge_index, edge_attr

    return remove_self_loops(edge_index, edge_attr)


def remove_self_loops(edge_index, edge_attr):
    """
    Removes all self-loop in the graph.

    Args:
        edge_index (`torch.LongTensor`): indices of graph's edges.
        edge_attr (`torch.FloatTensor`): attributes of graph's edges.

    Returns:
        (`torch.LongTensor`): indices of graph's edges.
        (`torch.FloatTensor`): attributes of graph's edges.
    """
    mask = edge_index[0] != edge_index[1]

    return edge_index[..., mask], edge_attr[..., mask]


def add_self_loops(edge_index, edge_attr, weight=1.0):
    """
    Add a self-loop for each vertex of the graph.

    Args:
        edge_index (`torch.LongTensor`): indices of graph's edges.
        edge_attr (`torch.FloatTensor`): attributes of graph's edges.
        weight (float, optional): weight of a self-loop. Defaults to 1.0.

    Returns:
        (`torch.LongTensor`): indices of graph's edges.
        (`torch.FloatTensor`): attributes of graph's edges.
    """

    self_loop_index = edge_index[0].unique().unsqueeze(0).repeat(2, 1)
    self_loop_attr = weight * torch.ones(self_loop_index.shape[1])

    edge_index = torch.cat((self_loop_index, edge_index), dim=1)
    edge_attr = torch.cat((self_loop_attr, edge_attr))

    return edge_index, edge_attr
