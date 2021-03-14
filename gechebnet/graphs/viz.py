# coding=utf-8

import pandas as pd
import plotly.express as px


def visualize_graph(graph):
    """
    Visualize graph's vertices.

    Args:
        graph (`Graph`): graph.
    """

    df = pd.DataFrame({"node_index": graph.node_index})
    df["X"], df["Y"], df["Z"] = graph.cartesian_pos()
    for attr in graph.node_attributes:
        df[attr] = getattr(graph, attr)

    fig = px.scatter_3d(df, x="X", y="Y", z="Z", hover_data=list(graph.node_attributes) + ["node_index"])

    fig.update_traces(
        marker={"size": 5, "color": "crimson", "line": {"width": 2, "color": "DarkSlateGrey"}, "opacity": 1.0},
    )

    fig.update_layout(width=500, height=500, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.show()


def visualize_graph_signal(graph, signal):
    """
    Visualize a signal on the graph's vertices.

    Args:
        graph (`Graph`): graph.
        signal (`torch.Tensor`): signal on the graph's vertices.
    """

    df = pd.DataFrame({"node_index": graph.node_index})
    df["X"], df["Y"], df["Z"] = graph.cartesian_pos()
    df["signal"] = signal
    for attr in graph.node_attributes:
        df[attr] = getattr(graph, attr)

    fig = px.scatter_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        color="signal",
        hover_data=list(graph.node_attributes) + ["signal", "node_index"],
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


def visualize_graph_neighborhood(graph, node_index):
    """
    Visualize graph neighborhood of the given node.

    Args:
        graph (`Graph`): graph.
        node_index (int): node index.
    """
    df1 = pd.DataFrame()
    df1["node_index"], df1["weight"], df1["sqdist"] = graph.neighborhood(node_index)

    df2 = pd.DataFrame({"node_index": graph.node_index})
    df2["X"], df2["Y"], df2["Z"] = graph.cartesian_pos()

    for attr in graph.node_attributes:
        df2[attr] = getattr(graph, attr)

    df = pd.merge(df1, df2, on="node_index", how="right")
    df.weight.fillna(0.0, inplace=True)

    fig = px.scatter_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        color="weight",
        hover_data=list(graph.node_attributes) + ["weight", "sqdist", "node_index"],
        color_continuous_scale="PuRd",
        range_color=[0, 1],
    )

    fig.update_traces(
        marker={"size": 5, "opacity": 1.0},
    )

    fig.update_layout(width=600, height=500, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.show()
