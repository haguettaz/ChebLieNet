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
