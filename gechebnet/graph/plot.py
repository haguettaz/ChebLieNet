from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from gechebnet.graph import Graph
from matplotlib.figure import Figure
from torch import FloatTensor

from ..utils import random_choice, rescale
from .signal_processing import get_fourier_basis


def visualize_graph(graph: Graph, signal: Optional[FloatTensor] = None) -> Figure:
    """
    Visualize graph with or without signal on it.

    Args:
        graph (Graph): graph.
        signal (FloatTensor, optional): graph's signal. Defaults to None.

    Returns:
        Figure: figure with graph visualization.
    """

    fig = plt.figure(figsize=(8.0, 8.0))

    ax = fig.add_subplot(
        111,
        projection="3d",
        xlim=(graph.x1_axis.min(), graph.x1_axis.max()),
        ylim=(graph.x2_axis.min(), graph.x2_axis.max()),
        zlim=(graph.x3_axis.min(), graph.x3_axis.max()),
    )

    if signal is None:
        ax.scatter(
            graph.node_pos[graph.node_index, 0],
            graph.node_pos[graph.node_index, 1],
            graph.node_pos[graph.node_index, 2],
            s=50,
            alpha=0.5,
        )

        return fig

    if signal.max() > 1 or signal.min < 0:
        signal = rescale(signal)

    ax.scatter(
        graph.node_pos[graph.node_index, 0],
        graph.node_pos[graph.node_index, 1],
        graph.node_pos[graph.node_index, 2],
        s=50,
        c=signal,
        alpha=0.5,
    )

    return fig


def visualize_neighborhood(graph: Graph, node_idx: int) -> Figure:
    """
    Visualize graph neighborhood of node index.

    Args:
        graph (Graph): graph.
        signal (FloatTensor, optional): graph's signal. Defaults to None.

    Returns:
        Figure: figure with graph visualization.
    """

    fig = plt.figure(figsize=(8.0, 8.0))

    ax = fig.add_subplot(
        111,
        projection="3d",
        xlim=(graph.x1_axis.min(), graph.x1_axis.max()),
        ylim=(graph.x2_axis.min(), graph.x2_axis.max()),
        zlim=(graph.x3_axis.min(), graph.x3_axis.max()),
    )

    neighbors_index, weights = graph.neighborhood(node_idx, return_weights=True)

    im = ax.scatter(
        graph.node_pos[neighbors_index, 0],
        graph.node_pos[neighbors_index, 1],
        graph.node_pos[neighbors_index, 2],
        s=50,
        c=weights,
        alpha=0.5,
    )

    plt.colorbar(im, fraction=0.04, pad=0.1)

    ax.scatter(
        graph.node_pos[node_idx, 0],
        graph.node_pos[node_idx, 1],
        graph.node_pos[node_idx, 2],
        s=50,
        c="white",
        edgecolors="black",
        linewidth=3,
        alpha=1.0,
    )

    return fig


def visualize_heat_diffusion(graph: Graph, f0: np.array, times: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)) -> Figure:
    """
    Visualize heat diffusion on graph.

    Args:
        graph (Graph): graph
        f0 (np.array): initial function.
        times (Tuple[float, ...], optional): diffusion times. Defaults to (0.0, 0.1, 0.2, 0.4).

    Returns:
        Figure: [description]
    """

    num_cols = len(times)
    fig = plt.figure(figsize=(num_cols * 8.0, 8.0))

    lambdas, Phi = graph.fourier_basis
    eps = 1e-6

    for c in range(num_cols):
        ft = Phi @ np.diag(np.exp(times[c] * lambdas)) @ Phi.T @ f0
        mask_nonzeros = np.abs(ft) > eps

        ax = fig.add_subplot(
            1,
            num_cols,
            c + 1,
            projection="3d",
            xlim=(graph.x1_axis.min(), graph.x1_axis.max()),
            ylim=(graph.x2_axis.min(), graph.x2_axis.max()),
            zlim=(graph.x3_axis.min(), graph.x3_axis.max()),
        )

        ax.scatter(
            graph.node_pos[graph.node_index, 0][mask_nonzeros],
            graph.node_pos[graph.node_index, 1][mask_nonzeros],
            graph.node_pos[graph.node_index, 2][mask_nonzeros],
            c=ft[mask_nonzeros],
            s=50,
            alpha=0.5,
        )

        ax.set_title(fr"heat diffusion at $t = {times[c]}$")

    return fig
