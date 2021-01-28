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


def visualize_graph(
    graph: Graph,
    signal: Optional[FloatTensor] = None,
    view_init: Optional[Tuple[float, float]] = None,
    show_axis: Optional[bool] = True,
) -> Figure:
    """
    Visualize graph with or without signal on it.

    Args:
        graph (Graph): graph.
        signal (FloatTensor, optional): graph's signal. Defaults to None.

    Returns:
        Figure: figure with graph visualization.
    """

    fig = plt.figure(figsize=(9.0, 8.0))

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    if view_init is not None:
        ax.view_init(*view_init)

    x, y, z = graph.node_pos

    if signal is None:
        im = ax.scatter(
            x[graph.node_index],
            y[graph.node_index],
            z[graph.node_index],
            s=50,
            alpha=0.5,
        )

    else:
        im = ax.scatter(
            x[graph.node_index],
            y[graph.node_index],
            z[graph.node_index],
            s=50,
            c=rescale(signal, 0.0, 1.0),
            alpha=0.5,
        )

    plt.colorbar(im, fraction=0.04, pad=0.1)

    plt.axis("on" if show_axis else "off")

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

    fig = plt.figure(figsize=(9.0, 8.0))

    ax = fig.add_subplot(111, projection="3d")

    neighbors_index, weights = graph.neighborhood(node_idx)

    x, y, z = graph.node_pos

    im = ax.scatter(
        x[neighbors_index],
        y[neighbors_index],
        z[neighbors_index],
        s=50,
        c=weights,
        alpha=0.5,
    )

    plt.colorbar(im, fraction=0.04, pad=0.1)

    ax.scatter(
        x[node_idx],
        y[node_idx],
        z[node_idx],
        s=50,
        c="white",
        edgecolors="black",
        linewidth=3,
        alpha=1.0,
    )

    return fig


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

    x, y, z = graph.node_pos

    fig = plt.figure(figsize=(9.0, 8.0))

    ax = fig.add_subplot(111, projection="3d")

    def update(t):
        ft = graph.diff_kernel(lambda x: diff_kernel(x, t)) @ f0
        mask_nonzeros = np.abs(ft) > tol

        ax.scatter(
            x[graph.node_index][mask_nonzeros],
            y[graph.node_index][mask_nonzeros],
            z[graph.node_index][mask_nonzeros],
            c=ft[mask_nonzeros],
            s=50,
            alpha=0.5,
        )

    ani = FuncAnimation(fig, update, times)
    writer = PillowWriter(fps=5)

    ani.save(file_name, writer=writer)
