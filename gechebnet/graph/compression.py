from torch import BoolTensor, FloatTensor, LongTensor

from ..utils import shuffle_tensor


def edge_compression(edge_index, edge_attr, kappa):
    """
    Randomly remove a given rate of edges from the original tensor of edge's indices.

    Args:
        edge_index (LongTensor): the original edge's indices.
        edge_attr (FloatTensor): the original edge's weights.
        kappa (float): the rate of edges to remove.

    Returns:
        (LongTensor): the compressed edge's indices.
        (FloatTensor): the compressed edge's weights.
    """
    num_to_remove = int(kappa * edge_index.shape[1])
    num_to_keep = edge_index.shape[1] - num_to_remove

    mask = BoolTensor([True] * num_to_remove + [False] * num_to_keep)
    mask = shuffle_tensor(mask)

    return edge_index[:, ~mask], edge_attr[~mask]
