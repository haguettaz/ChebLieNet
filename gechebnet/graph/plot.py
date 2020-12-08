import matplotlib.pyplot as plt
import torch

from ..utils import random_choice
from .graph import get_neighbors


def visualize_graph(graph_data):
    """
    3d visualization of the nodes of the graph in addition to the number of nodes and number of edges

    Args:
        graph_data (GraphData): the GraphData object containing the graph.
    """
    fig = plt.figure(figsize=(8.0, 8.0))

    ax = fig.add_subplot(
        111,
        projection="3d",
        xlim=(graph_data.x1_axis.min(), graph_data.x1_axis.max()),
        ylim=(graph_data.x2_axis.min(), graph_data.x2_axis.max()),
        zlim=(graph_data.x3_axis.min(), graph_data.x3_axis.max()),
        title=f"{graph_data.data.num_nodes} nodes - {graph_data.data.num_edges} edges",
    )

    ax.scatter(
        graph_data.node_pos[graph_data.node_index, 0],
        graph_data.node_pos[graph_data.node_index, 1],
        graph_data.node_pos[graph_data.node_index, 2],
        s=50,
        alpha=0.5,
    )

    return fig


def visualize_weight_fields(graph_data, grid_size=(2, 2)):
    """
    3d visualizations of the weight field from randomly picked nodes of the graph.

    Args:
        graph_data (GraphData): the GraphData object containing the graph.
        grid_size (tuple, optional): the size of the grid containing the 3d visualization in format
            (num_rows, num_cols). The total number of visualization is num_rows * num_cols. Defaults
            to (2, 2).
    """
    num_rows, num_cols = grid_size

    fig = plt.figure(figsize=(num_rows * 8.0, num_cols * 8.0))

    for r in range(num_rows):
        for c in range(num_cols):
            node_idx = random_choice(graph_data.node_index)
            neighbors, weights = get_neighbors(graph_data, node_idx)

            ax = fig.add_subplot(
                num_rows,
                num_cols,
                r * num_cols + c + 1,
                projection="3d",
                xlim=(graph_data.x1_axis.min(), graph_data.x1_axis.max()),
                ylim=(graph_data.x2_axis.min(), graph_data.x2_axis.max()),
                zlim=(graph_data.x3_axis.min(), graph_data.x3_axis.max()),
            )

            im = ax.scatter(
                graph_data.node_pos[neighbors, 0], graph_data.node_pos[neighbors, 1], graph_data.node_pos[neighbors, 2], c=weights, s=50, alpha=0.5
            )

            plt.colorbar(im, fraction=0.04, pad=0.1)
            im = ax.scatter(
                graph_data.node_pos[node_idx, 0],
                graph_data.node_pos[node_idx, 1],
                graph_data.node_pos[node_idx, 2],
                s=50,
                c="white",
                edgecolors="black",
                linewidth=3,
                alpha=1.0,
            )

            ax.set_title(f"node #{node_idx.item()}")

    return fig


def visualize_weight_field(graph_data):
    """
    3d visualizations of the weight field from randomly picked nodes of the graph.

    Args:
        graph_data (GraphData): the GraphData object containing the graph.
    """
    fig = plt.figure(figsize=(8.0, 8.0))

    node_idx = random_choice(graph_data.node_index)
    neighbors, weights = get_neighbors(graph_data, node_idx)

    ax = fig.add_subplot(
        111,
        projection="3d",
        xlim=(graph_data.x1_axis.min(), graph_data.x1_axis.max()),
        ylim=(graph_data.x2_axis.min(), graph_data.x2_axis.max()),
        zlim=(graph_data.x3_axis.min(), graph_data.x3_axis.max()),
    )

    ax.scatter(graph_data.node_pos[neighbors, 0], graph_data.node_pos[neighbors, 1], graph_data.node_pos[neighbors, 2], c=weights, s=50, alpha=0.5)

    ax.set_title(f"node #{node_idx.item()}")

    return fig


def visualize_samples(data_list, grid_size=(3, 3)):
    """
    3d visualization of randomly picked sample from the list of Data object.

    Args:
        data_list (list): the list of Data object
        grid_size (tuple, optional): the size of the grid containing the 3d visualization in format
            (num_rows, num_cols). The total number of visualization is num_rows * num_cols. Defaults
            to (3, 3).
    """
    num_rows, num_cols = grid_size

    fig = plt.figure(figsize=(num_rows * 8.0, num_cols * 8.0))

    for r in range(num_rows):
        for c in range(num_cols):
            sample_idx = random_choice(torch.arange(len(data_list))).item()
            sample = data_list[sample_idx]
            max_, _ = sample.x.max(dim=0)
            min_, _ = sample.x.min(dim=0)

            ax = fig.add_subplot(num_rows, num_cols, r * num_cols + c + 1, projection="3d")

            ax.scatter(
                sample.pos[:, 0],
                sample.pos[:, 1],
                sample.pos[:, 2],
                c=(sample.x - min_) / (max_ - min_),
                alpha=0.5,
            )

            ax.set_title(f"sample #{sample_idx} labeled {sample.y.item()}")

    return fig


def visualize_signal(graph_data, signal):
    """
    3d visualization of a signal on the graph.

    Args:
        graph_data (GraphData): the GraphData object containing the graph.
        signal (torch.tensor): the signal on the graph to plot
    """
    fig = plt.figure(figsize=(8.0, 8.0))

    ax = fig.add_subplot(
        111,
        projection="3d",
        xlim=(graph_data.x1_axis.min(), graph_data.x1_axis.max()),
        ylim=(graph_data.x2_axis.min(), graph_data.x2_axis.max()),
        zlim=(graph_data.x3_axis.min(), graph_data.x3_axis.max()),
    )

    im = ax.scatter(
        graph_data.node_pos[graph_data.node_index, 0],
        graph_data.node_pos[graph_data.node_index, 1],
        graph_data.node_pos[graph_data.node_index, 2],
        c=signal,
        s=50,
        alpha=0.5,
    )

    plt.colorbar(im, fraction=0.04, pad=0.1)

    return fig
