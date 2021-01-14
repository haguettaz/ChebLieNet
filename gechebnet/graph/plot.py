from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from numpy import ndarray
from torch import FloatTensor

from ..utils import random_choice, rescale
from .graph import Graph
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
        im = ax.scatter(
            graph.node_pos[graph.node_index, 0],
            graph.node_pos[graph.node_index, 1],
            graph.node_pos[graph.node_index, 2],
            s=50,
            alpha=0.5,
        )

    else:
        if signal.max() > 1 or signal.min() < 0:
            signal = rescale(signal)

        im = ax.scatter(
            graph.node_pos[graph.node_index, 0],
            graph.node_pos[graph.node_index, 1],
            graph.node_pos[graph.node_index, 2],
            s=50,
            c=signal,
            alpha=0.5,
        )

    plt.colorbar(im, fraction=0.04, pad=0.1)

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

    neighbors_index, weights = graph.neighborhood(node_idx)

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


def visualize_heat_diffusion(
    graph: Graph,
    f0: ndarray = None,
    times: ndarray = None,
    tol: float = 1e-6,
    file_name: str = "heat_diffusion.gif",
):
    """
    Visualize heat diffusion on graph by creating a gif.

    Args:
        graph (Graph): graph.
        f0 (ndarray): initial function. Defaults to None.
        times (ndarray, optional): diffusion times. Defaults to None.
        tol (float, optional): tolerance to detect zero values. Defaults to 1e-6.
        file_name (str, optional): name of the generated gif. Defaults to 'heat_diffusion.gif'
    """

    if f0 is None:
        f0 = graph.dirac(graph.centroid_index)

    if times is None:
        times = np.arange(0.0, 1.0, 0.1)

    fig = plt.figure(figsize=(8.0, 8.0))

    ax = fig.add_subplot(
        111,
        projection="3d",
        xlim=(graph.x1_axis.min(), graph.x1_axis.max()),
        ylim=(graph.x2_axis.min(), graph.x2_axis.max()),
        zlim=(graph.x3_axis.min(), graph.x3_axis.max()),
    )

    def update(t):
        ft = graph.heat_kernel(t) @ f0
        mask_nonzeros = np.abs(ft) > tol

        ax.scatter(
            graph.node_pos[graph.node_index, 0][mask_nonzeros],
            graph.node_pos[graph.node_index, 1][mask_nonzeros],
            graph.node_pos[graph.node_index, 2][mask_nonzeros],
            c=ft[mask_nonzeros],
            s=50,
            alpha=0.5,
        )

    ani = FuncAnimation(fig, update, times)
    writer = PillowWriter(fps=5)

    ani.save(file_name, writer=writer)


def visualize_diffusion_process(
    graph: Graph,
    f0: ndarray = None,
    times: ndarray = None,
    diff_kernel: Callable = None,
    tol: float = 1e-6,
    file_name: str = "diffusion.gif",
):
    """
    Visualize heat diffusion on graph by creating a gif.

    Args:
        graph (Graph): graph.
        f0 (ndarray): initial function. Defaults to None.
        times (ndarray, optional): diffusion times. Defaults to None.
        diff_kernel (callable, optional): diffusion kernel taking as input x and t and returning y. Defaults to None.
        tol (float, optional): tolerance to detect zero values. Defaults to 1e-6.
        file_name (str, optional): name of the generated gif. Defaults to 'heat_diffusion.gif'
    """

    if f0 is None:
        f0 = graph.dirac(graph.centroid_index)

    if times is None:
        times = np.arange(0.0, 1.0, 0.1)

    if diff_kernel is None:
        diff_kernel = lambda x, t: np.power(x, t)

    fig = plt.figure(figsize=(8.0, 8.0))

    ax = fig.add_subplot(
        111,
        projection="3d",
        xlim=(graph.x1_axis.min(), graph.x1_axis.max()),
        ylim=(graph.x2_axis.min(), graph.x2_axis.max()),
        zlim=(graph.x3_axis.min(), graph.x3_axis.max()),
    )

    def update(t):
        ft = graph.diff_kernel(lambda x: diff_kernel(x, t)) @ f0

        ax.scatter(
            graph.node_pos[graph.node_index, 0],
            graph.node_pos[graph.node_index, 1],
            graph.node_pos[graph.node_index, 2],
            c=ft,
            s=50,
        )

    ani = FuncAnimation(fig, update, times)
    writer = PillowWriter(fps=5)

    ani.save(file_name, writer=writer)
