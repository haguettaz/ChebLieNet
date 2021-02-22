from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from numpy import ndarray
from torch import FloatTensor

from ..utils import random_choice, rescale
from .graph import Graph
from .signal_processing import get_fourier_basis


def visualize_graph(graph: Graph):
    """
    Visualize graph's vertices.

    Args:
        graph (Graph): graph.
    """
    if graph.lie_group == "se2":
        dataframe = pd.DataFrame(
            {
                "X": graph.node_pos("x"),
                "Y": graph.node_pos("y"),
                "Z": graph.node_pos("z"),
                "x": graph.node_x,
                "y": graph.node_y,
                "theta": graph.node_theta,
            }
        )
        fig = px.scatter_3d(dataframe, x="X", y="Y", z="Z", hover_data=["x", "y", "theta"])

    elif graph.lie_group == "so3":
        dataframe = pd.DataFrame(
            {
                "X": graph.node_pos("x"),
                "Y": graph.node_pos("y"),
                "Z": graph.node_pos("z"),
                "alpha": graph.node_alpha,
                "beta": graph.node_beta,
                "gamma": graph.node_gamma,
            }
        )
        fig = px.scatter_3d(dataframe, x="X", y="Y", z="Z", hover_data=["alpha", "beta", "gamma"])

    fig.update_traces(
        marker={"size": 5, "color": "crimson", "line": {"width": 2, "color": "DarkSlateGrey"}, "opacity": 1.0},
    )

    fig.update_layout(width=500, height=500, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.show()


def visualize_graph_signal(graph: Graph, signal: Tensor):
    """
    Visualize graph's signal.

    Args:
        graph (Graph): graph.
        signal (Tensor): graph's signal.
    """
    if graph.lie_group == "se2":
        dataframe = pd.DataFrame(
            {
                "X": graph.node_pos("x"),
                "Y": graph.node_pos("y"),
                "Z": graph.node_pos("z"),
                "x": graph.node_x,
                "y": graph.node_y,
                "theta": graph.node_theta,
                "intensity": signal,
            }
        )
        fig = px.scatter_3d(
            dataframe,
            x="X",
            y="Y",
            z="Z",
            color="intensity",
            hover_data=["x", "y", "theta", "intensity"],
            color_continuous_scale="PiYG",
            color_continuous_midpoint=0.0,
        )

    elif graph.lie_group == "so3":
        dataframe = pd.DataFrame(
            {
                "X": graph.node_pos("x"),
                "Y": graph.node_pos("y"),
                "Z": graph.node_pos("z"),
                "alpha": graph.node_alpha,
                "beta": graph.node_beta,
                "gamma": graph.node_gamma,
                "intensity": signal,
            }
        )
        fig = px.scatter_3d(
            dataframe,
            x="X",
            y="Y",
            z="Z",
            color="intensity",
            hover_data=["alpha", "beta", "gamma", "intensity"],
            color_continuous_scale="PiYG",
            color_continuous_midpoint=0.0,
        )

    fig.update_traces(
        marker={"size": 5, "opacity": 1.0},
    )

    fig.update_layout(
        width=600,
        height=500,
        margin=dict(l=0, r=0, t=0, b=50),
    )
    fig.show()


def visualize_graph_neighborhood(graph: Graph, node_index: int):
    """
    Visualize graph neighborhood of the given node.

    Args:
        graph (Graph): graph.
        node_index (int): node index.
    """
    neighborhood_index, neighborhood_weight, neighborhood_sqdist = graph.neighborhood(node_index)
    weight = torch.zeros(graph.num_nodes)
    sqdist = torch.zeros(graph.num_nodes)
    weight[neighborhood_index] = neighborhood_weight
    sqdist[neighborhood_index] = neighborhood_sqdist

    if graph.lie_group == "se2":
        dataframe = pd.DataFrame(
            {
                "X": graph.node_pos("x"),
                "Y": graph.node_pos("y"),
                "Z": graph.node_pos("z"),
                "x": graph.node_x,
                "y": graph.node_y,
                "theta": graph.node_theta,
                "sqdist": sqdist,
                "weight": weight,
            }
        )
        fig = px.scatter_3d(
            dataframe,
            x="X",
            y="Y",
            z="Z",
            color="weight",
            hover_data=["x", "y", "theta", "sqdist", "weight"],
            color_continuous_scale="PuRd",
            range_color=[0, 1],
        )
    elif graph.lie_group == "so3":
        dataframe = pd.DataFrame(
            {
                "X": graph.node_pos("x"),
                "Y": graph.node_pos("y"),
                "Z": graph.node_pos("z"),
                "alpha": graph.node_alpha,
                "beta": graph.node_beta,
                "gamma": graph.node_gamma,
                "sqdist": sqdist,
                "weight": weight,
            }
        )
        fig = px.scatter_3d(
            dataframe,
            x="X",
            y="Y",
            z="Z",
            color="weight",
            hover_data=["alpha", "beta", "gamma", "sqdist", "weight"],
            color_continuous_scale="PuRd",
        )

    fig.update_traces(
        marker={"size": 5, "opacity": 1.0},
    )

    fig.update_layout(width=600, height=500, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.show()