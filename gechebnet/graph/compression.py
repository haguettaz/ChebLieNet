import torch

from ..utils import shuffle_tensor


def node_compression(graph, kappa):
    """
    Randomly remove a given rate of nodes from the original tensor of node's indices

    Args:
        node_index (torch.tensor): the original node's indices.
        kappa (float): the rate of nodes to remove.

    Returns:
        (torch.tensor): the compressed node's indices.
    """

    num_to_remove = int(kappa * graph.num_nodes)
    num_to_keep = graph.num_nodes - num_to_remove

    node_mask = torch.tensor([True] * num_to_remove + [False] * num_to_keep)
    node_mask = shuffle(node_mask)

    edge_mask = torch.tensor([False] * graph.num_edges)

    for n_idx in graph.node_index[node_mask]:
        edge_mask |= (graph.edge_index[0] == n_idx) | (graph.edge_index[1] == n_idx)

    graph.node_index = graph.node_index[~node_mask]
    graph.edge_index = graph.edge_index[:, ~edge_mask]
    graph.edge_weight = graph.edge_weight[~edge_mask]

    return graph


def edge_compression(graph, kappa):
    """
    Randomly remove a given rate of edges from the original tensor of edge's indices.

    Args:
        edge_index (torch.tensor): the original edge's indices.
        edge_weight (torch.tensor): the original edge's weights.
        kappa (float): the rate of edges to remove.

    Returns:
        (torch.tensor): the compressed edge's indices.
        (torch.tensor): the compressed edge's weights.
    """

    num_to_remove = int(kappa * graph.num_edges)
    num_to_keep = graph.num_edges - num_to_remove

    mask = torch.tensor([True] * num_to_remove + [False] * num_to_keep)
    mask = shuffle(mask)

    graph.edge_index = graph.edge_index[:, ~mask]
    graph.edge_weight = graph.edge_weight[~mask]

    return graph
