import matplotlib.pyplot as plt
import torch

from ..utils.utils import random_choice
from .graph import get_neighbors


def visualize_weight_fields(graph_data, grid_size=(2, 2)):
    """
    [summary]

    Args:
        graph_data ([type]): [description]
        grid_size (tuple, optional): [description]. Defaults to (2, 2).
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
                xlim=(graph_data.x_axis.min(), graph_data.x_axis.max()),
                ylim=(graph_data.y_axis.min(), graph_data.y_axis.max()),
                zlim=(graph_data.z_axis.min(), graph_data.z_axis.max()),
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

    plt.show()


def visualize_samples(datalist, grid_size=(3, 3)):
    """
    [summary]

    Args:
        datalist ([type]): [description]
        grid_size (tuple, optional): [description]. Defaults to (3,3).
    """
    num_rows, num_cols = grid_size

    fig = plt.figure(figsize=(num_rows * 8.0, num_cols * 8.0))

    for r in range(num_rows):
        for c in range(num_cols):
            sample_idx = random_choice(torch.arange(len(datalist))).item()
            sample = datalist[sample_idx]
            ax = fig.add_subplot(num_rows, num_cols, r * num_cols + c + 1, projection="3d")

            im = ax.scatter(sample.pos[:, 0], sample.pos[:, 1], sample.pos[:, 2], c=sample.x, s=50, alpha=0.5)

            plt.colorbar(im, fraction=0.04, pad=0.1)

            ax.set_title(f"sample #{sample_idx} labeled {sample.y.item()}")

    plt.show()
