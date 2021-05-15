# coding=utf-8

import torch


def to_undirected(edge_index, edge_sqdist, num_nodes, max_sqdist, self_loop=False):
    """
    Make the graph undirected, that is create an inverse edge for each edge.

    Args:
        edge_index (`torch.LongTensor`): indices of vertices connected by the edges of the graph.
        edge_sqdist (`torch.FloatTensor`): squared distances encoded on the edges of the graph.
        num_nodes (int): number of vertices of the graph.
        max_sqdist (float): maximum squared distance between two connected vertices.
        self_loop (bool): indicator wether the graph contains self loop.

    Returns:
        (`torch.LongTensor`): indices of graph's edges.
        (`torch.FloatTensor`): attributes of graph's edges.
    """
    sqdist_matrix = torch.sparse.FloatTensor(edge_index, edge_sqdist, torch.Size((num_nodes, num_nodes))).to_dense()

    mask = (sqdist_matrix.t() == sqdist_matrix) & (sqdist_matrix <= max_sqdist)

    undirected_sqdist_matrix = torch.zeros_like(sqdist_matrix)
    undirected_sqdist_matrix[mask] = sqdist_matrix[mask]
    undirected_sqdist_matrix = undirected_sqdist_matrix.to_sparse()

    edge_index = undirected_sqdist_matrix.indices()
    edge_sqdist = undirected_sqdist_matrix.values()

    if self_loop:
        return edge_index, edge_sqdist

    return remove_self_loops(edge_index, edge_sqdist)


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
