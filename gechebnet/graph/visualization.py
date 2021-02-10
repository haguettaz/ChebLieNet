from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from numpy import ndarray
from torch import FloatTensor

from ..utils import random_choice, rescale
from .graph import Graph
from .signal_processing import get_fourier_basis


def visualize_graph(graph):
    data = [
        go.Scatter3d(
            x=graph.node_pos("x"),
            y=graph.node_pos("y"),
            z=graph.node_pos("z"),
            mode="markers",
            marker=dict(
                size=5,
                color="firebrick",
                opacity=0.8,
            ),
        )
    ]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def visualize_graph_signal(graph, signal):
    data = [
        go.Scatter3d(
            x=graph.node_pos("x"),
            y=graph.node_pos("y"),
            z=graph.node_pos("z"),
            mode="markers",
            marker=dict(
                size=5,
                color=signal,
                opacity=0.8,
            ),
        )
    ]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def visualize_graph_neighborhood(graph, node_index):
    data = [
        go.Scatter3d(
            x=graph.node_pos("x"),
            y=graph.node_pos("y"),
            z=graph.node_pos("z"),
            mode="markers",
            marker=dict(
                size=5,
                color=graph.neighbors_signal(node_index),
                opacity=0.8,
            ),
        )
    ]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

    fig = go.Figure(data=data, layout=layout)
    fig.show()


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
