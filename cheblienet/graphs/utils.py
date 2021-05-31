# coding=utf-8

import torch


def to_undirected(edge_index, edge_sqdist, edge_weight=None, num_vertices=None, max_sqdist=None, self_loop=False):
    """
    Make the graph undirected, that is create an inverse edge for each edge.

    Args:
        edge_index (`torch.LongTensor`): indices of vertices connected by the edges of the graph.
        edge_sqdist (`torch.FloatTensor`): squared distances encoded on the edges of the graph.
        num_vertices (int): number of vertices of the graph.
        max_sqdist (float): maximum squared distance between two connected vertices.
        self_loop (bool): indicator wether the graph contains self loop.

    Returns:
        (`torch.LongTensor`): indices of graph's edges.
        (`torch.FloatTensor`): attributes of graph's edges.
    """
    num_vertices = num_vertices or edge_index.max() + 1

    sqdist_matrix = torch.sparse.FloatTensor(edge_index, edge_sqdist, torch.Size((num_vertices, num_vertices))).to_dense()

    mask = sqdist_matrix.t() == sqdist_matrix
    if max_sqdist is not None:
        mask &= sqdist_matrix <= max_sqdist

    undirected_sqdist_matrix = torch.zeros_like(sqdist_matrix)
    undirected_sqdist_matrix[mask] = sqdist_matrix[mask]
    undirected_sqdist_matrix = undirected_sqdist_matrix.to_sparse()

    if edge_weight is not None:
        weight_matrix = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size((num_vertices, num_vertices))).to_dense()
        undirected_weight_matrix = torch.zeros_like(weight_matrix)
        undirected_weight_matrix[mask] = weight_matrix[mask]
        undirected_weight_matrix = undirected_weight_matrix.to_sparse()

    edge_index = undirected_sqdist_matrix.indices()
    edge_sqdist = undirected_sqdist_matrix.values()

    if edge_weight is not None:
        edge_weight = undirected_weight_matrix.values()

    if self_loop:
        if edge_weight is None:
            return edge_index, edge_sqdist

        return edge_index, edge_sqdist, edge_weight

    return remove_self_loops(edge_index, edge_sqdist, edge_weight)


def remove_self_loops(edge_index, edge_sqdist, edge_weight=None):
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

    if edge_weight is None:
        return edge_index[..., mask], edge_sqdist[mask]

    return edge_index[..., mask], edge_sqdist[mask], edge_weight[mask]


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
