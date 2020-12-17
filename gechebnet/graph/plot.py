import matplotlib.pyplot as plt
import numpy as np
import torch

from ..utils import normalize, random_choice
from .signal_processing import get_fourier_basis


def visualize_graph(graph, signal=None):
    """
    3d visualization of the nodes of the graph in addition to the number of nodes and number of edges

    Args:
        graph_data (GraphData): the GraphData object containing the graph.
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

    if torch.max(signal) > 1 or torch.min(signal) < 0:
        signal = normalize(signal)

    ax.scatter(
        graph.node_pos[graph.node_index, 0],
        graph.node_pos[graph.node_index, 1],
        graph.node_pos[graph.node_index, 2],
        s=50,
        c=signal,
        alpha=0.5,
    )

    return fig


def visualize_neighborhood(graph, node_idx):
    """
    3d visualization of the nodes of the graph in addition to the number of nodes and number of edges

    Args:
        graph_data (GraphData): the GraphData object containing the graph.
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


def visualize_heat_diffusion(graph_data, f0, times=(0.0, 0.1, 0.2, 0.4), normalization=None):

    num_cols = len(times)
    fig = plt.figure(figsize=(num_cols * 8.0, 8.0))

    lambdas, Phi = compute_fourier_basis(graph_data, normalization)
    eps = 1e-9

    for c in range(num_cols):
        ft = Phi @ np.diag(np.exp(times[c] * lambdas)) @ Phi.T @ f0
        mask_nonzeros = [np.abs(ft) > eps]

        ax = fig.add_subplot(
            1,
            num_cols,
            c + 1,
            projection="3d",
            xlim=(graph_data.x1_axis.min(), graph_data.x1_axis.max()),
            ylim=(graph_data.x2_axis.min(), graph_data.x2_axis.max()),
            zlim=(graph_data.x3_axis.min(), graph_data.x3_axis.max()),
        )

        ax.scatter(
            graph_data.node_pos[graph_data.node_index, 0][mask_nonzeros],
            graph_data.node_pos[graph_data.node_index, 1][mask_nonzeros],
            graph_data.node_pos[graph_data.node_index, 2][mask_nonzeros],
            c=ft[mask_nonzeros],
            s=50,
            alpha=0.5,
        )

        ax.set_title(fr"heat diffusion at $t = {times[c]}$")

    return fig
