from typing import Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor

from ..utils import shuffle_tensor


def multinomial_compression(
    edge_index: LongTensor, edge_weight: FloatTensor, kappa: float
) -> Tuple[LongTensor, FloatTensor]:
    """
    Randomly sample a rate kappa of edges to remove from the graph.
    For the whole graph, samples from a multinomial distribution with probabilities
    the weights of the edges.

    Args:
        edge_index (LongTensor): the original edge's indices.
        edge_weight (FloatTensor): the original edge's weights.
        kappa (float): the rate of edges to remove.

    Returns:
        (LongTensor): the compressed edge's indices.
        (FloatTensor): the compressed edge's weights.
    """
    num_samples = int((1 - kappa) * edge_index.shape[1])
    mask = torch.multinomial(edge_weight, num_samples)
    return edge_index[:, mask], edge_weight[mask]


def bernoulli_compression(
    edge_index: LongTensor, edge_weight: FloatTensor, kappa: float
) -> Tuple[LongTensor, FloatTensor]:
    """
    Randomly sample a rate kappa of edges to remove from the graph.
    For each edge, samples from a bernoulli distribution with probability
    (1-kappa)*weight. A failure results in an edge dropping.

    Args:
        edge_index (LongTensor): the original edge's indices.
        edge_weight (FloatTensor): the original edge's weights.
        kappa (float): the rate of edges to remove.

    Returns:
        (LongTensor): the compressed edge's indices.
        (FloatTensor): the compressed edge's weights.
    """
    mask = torch.bernoulli((1 - kappa) * edge_weight).bool()
    return edge_index[:, mask], edge_weight[mask]
