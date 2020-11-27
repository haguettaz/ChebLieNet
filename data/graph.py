import math

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

from utils import metric_tensor

from .utils.utils import random_choice


class GraphData(object):
    def __init__(self, grid_size, num_layers=6, kappa=0.0, self_loop=True, sigma=1.0, weight_threshold=0.3, lambdas=(1.0, 1.0, 1.0)):
        """
        [summary]

        Args:
            grid_size ([type]): [description]
            num_layers (int, optional): [description]. Defaults to 6.
            kappa (float, optional): [description]. Defaults to 0.0.
            self_loop (bool, optional): [description]. Defaults to True.
            sigma (float, optional): [description]. Defaults to 1.0.
            weight_threshold (float, optional): [description]. Defaults to 0.3.
            lambdas (tuple, optional): [description]. Defaults to (1.0, 1.0, 1.0).
        """
        # nodes
        self.nx, self.ny = grid_size
        self.nz = num_layers
        self.init_nodes(self.nx * self.ny * self.nz, kappa)

        # edges
        self.self_loop = self_loop
        self.weight_threshold = weight_threshold
        self.sigma = sigma
        self.l1, self.l2, self.l3 = lambdas
        self.init_edges()

    def init_nodes(self, num_nodes, kappa):
        """
        [summary]

        Args:
            num_nodes ([type]): [description]
            kappa ([type]): [description]
        """
        node_index = torch.arange(num_nodes)

        # we define the grid points and reshape them to get 1-d arrays
        self.x_axis = torch.arange(0.0, self.nx, 1.0)
        self.y_axis = torch.arange(0.0, self.ny, 1.0)
        self.z_axis = torch.arange(0.0, math.pi, math.pi / self.nz)

        # we keep in memory the position of all the nodes, before compression
        # easier to deal with indices from here
        xv, yv, zv = torch.meshgrid(self.x_axis, self.y_axis, self.z_axis)
        self.node_pos = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)

        self.node_index = compression(node_index, kappa)
        self.num_nodes = self.node_index.nelement()

    def init_edges(self):
        """
        [summary]
        """
        distances = self.compute_distances()
        weights = self.compute_weights(distances)
        edge_indices = torch.reshape(torch.stack(torch.meshgrid(self.node_index, self.node_index), -1), [-1, 2])
        threshold_mask = weights >= self.weight_threshold

        self.edge_index = torch.transpose(edge_indices[threshold_mask], 1, 0)
        self.edge_metric = distances[threshold_mask]
        self.edge_weight = weights[threshold_mask]

    def compute_distances(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        distances = torch.zeros([self.num_nodes, self.num_nodes], dtype=torch.float32)
        difference_vectors = torch.cat(
            (
                self.node_pos[self.node_index, :2].unsqueeze(1) - self.node_pos[self.node_index, :2].unsqueeze(0),
                (
                    ((self.node_pos[self.node_index, 2].unsqueeze(1) - self.node_pos[self.node_index, 2].unsqueeze(0) - math.pi / 2) % math.pi)
                    - math.pi / 2
                ).unsqueeze(2),
            ),
            dim=2,
        )
        for z in self.z_axis:
            z_selection = self.node_pos[self.node_index, 2] == z
            dists = torch.matmul(
                difference_vectors[z_selection].unsqueeze(2),
                torch.matmul(metric_tensor(z, self.l1, self.l2, self.l3), difference_vectors[z_selection].unsqueeze(3)),
            )
            distances[z_selection, :] = dists[:, :, 0, 0]

        return distances.flatten()

    def compute_weights(self, distances):
        """
        [summary]

        Args:
            distances ([type]): [description]

        Returns:
            [type]: [description]
        """
        return torch.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def embed_on_graph(self, images, targets=None):
        """
        [summary]

        Args:
            images ([type]): [description]
            targets ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        num_images, channels, height, width = images.shape

        if height != self.ny or width != self.nx:
            raise ValueError("Impossible to embed images on the graph, the dimensions don't fit")

        images = images.unsqueeze(4).permute(0, 3, 2, 4, 1)
        images = images.expand(num_images, self.nx, self.ny, self.nz, channels)
        images = images.reshape(num_images, -1, channels)

        if targets is None:
            return [
                Data(x=images[idx, self.node_index], pos=self.node_pos[self.node_index], edge_index=self.edge_index, edge_attr=self.edge_weight)
                for idx in range(num_images)
            ]

        return [
            Data(
                x=images[idx, self.node_index],
                pos=self.node_pos[self.node_index],
                y=targets[idx],
                edge_index=self.edge_index,
                edge_attr=self.edge_weight,
            )
            for idx in range(num_images)
        ]

    @property
    def data(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return Data(edge_index=self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes)


def get_neighbors(graph_data, node_idx, return_weights=True):
    mask = (graph_data.edge_index[0] == node_idx) & (graph_data.edge_weight > 0.0)
    neighbors = graph_data.edge_index[1, mask]

    if not return_weights:
        return neighbors

    weights = graph_data.edge_weight[mask]
    return neighbors, weights


def visualize_weight_field(graph_data, grid_size=(2, 2)):
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
