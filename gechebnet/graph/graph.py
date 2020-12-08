import math

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

from ..utils import random_choice, shuffle
from .utils import CauchyKernel, GaussianKernel, WeightKernel, metric_tensor


class GraphData(object):
    def __init__(
        self,
        grid_size,
        num_layers=6,
        static_compression=None,
        self_loop=True,
        weight_kernel=None,
        sigmas=(1.0, 1.0, 1.0),
        batch_size=1000,
    ):
        """
        Initialise the GraphData object:
            - init the nodes and positions
            - init the edges and weights

        Args:
            grid_size (tuple): the spatial dimension of the graph in format (nx1, nx2).
            num_layers (int, optional): the orientation dimension of the graph. Defaults to 6.
            static_compression (tuple, optional): the static compression algorithm to reduce
                the graph size. Either ("node", kappa) or ("edge", kappa). Defaults to None.
            self_loop (bool, optional): the indicator if the graph can contains self-loop. Defaults to True.
            weight_kernel (WeightKernel, optional): the weights' kernel. Defaults to None.
            dist_threshold (float, optional): the maximum distance between two nodes to be linked. Defaults to 1.0.
            sigmas (tuple, optional): the anisotropic intensities. Defaults to (1.0, 1.0, 1.0).
            batch_size (int, optional): the batch size when computing edges' weights. Defaults to 1000.
        """

        # graph compression
        if static_compression is not None:
            self.compression_type, self.kappa = static_compression
        else:
            self.compression_type = None

        # nodes
        self.nx1, self.nx2 = grid_size
        self.nx3 = num_layers
        self.init_nodes(self.nx1 * self.nx2 * self.nx3)

        # edges
        self.self_loop = self_loop
        self.weight_kernel = weight_kernel or WeightKernel()
        self.sigmas = sigmas
        self.init_edges(batch_size)

    def init_nodes(self, num_nodes):
        """
        Initialise the nodes' indices and their positions.
            - `self.node_index` is a tensor with shape (num_nodes)
            - `self.node_pos` is a tensor with shape (self.nx1 * self.nx2 * self.nx3, 3)
        If the compression algorithm is the static node compression, remove a proportion kappa of nodes.

        Args:
            num_nodes (int): the number of nodes of the graph.
        """
        self.node_index = torch.arange(num_nodes)

        # we define the grid points and reshape them to get 1-d arrays
        self.x1_axis = torch.arange(0.0, self.nx1, 1.0)
        self.x2_axis = torch.arange(0.0, self.nx2, 1.0)
        self.x3_axis = torch.arange(0.0, math.pi, math.pi / self.nx3)

        # we keep in memory the position of all the nodes, before compression
        # easier to deal with indices from here
        x3_, x2_, x1_ = torch.meshgrid(self.x3_axis, self.x2_axis, self.x1_axis)
        self.node_pos = torch.stack([x1_.flatten(), x2_.flatten(), x3_.flatten()], axis=-1)

        if self.compression_type == "node":
            self.node_index = static_node_compression(self.node_index, self.kappa)

        self.num_nodes = self.node_index.nelement()

    def init_edges(self, batch_size):
        """
        Initialize the edges' indices and their weights.
            - `self.edge_index` is a tensor with shape (2, num_edges)
            - `self.edge_weight` is a tensor with shape (num_edges)

        If the compression algorithm is the static edge compression, remove a proportion kappa of edges.

        Args:
            batch_size (int): the size of a batch when computing distances between nodes
        """
        list_of_edges = []
        list_of_weights = []

        for batch in torch.split(self.node_index, batch_size):
            edge_index = torch.stack((batch.repeat_interleave(self.num_nodes), self.node_index.repeat(len(batch))))

            sq_dist = self.compute_sq_dist(self.node_pos[edge_index[0]], self.node_pos[edge_index[1]])

            edge_weights = self.weight_kernel.compute(sq_dist)
            nonzero_mask = torch.nonzero(edge_weights).flatten()

            list_of_edges.append(edge_index[:, nonzero_mask])
            list_of_weights.append(edge_weights[nonzero_mask])

        self.edge_index = torch.cat(list_of_edges, axis=1)
        self.edge_weight = torch.cat(list_of_weights)

        if not self.self_loop:
            self_loop_mask = self.edge_index[0] != self.edge_index[1]
            self.edge_index = self.edge_index[:, self_loop_mask]
            self.edge_weight = self.edge_weight[self_loop_mask]

        if self.compression_type == "edge":
            self.edge_index, self.edge_weight = static_edge_compression(self.edge_index, self.edge_weight, self.kappa)

    def compute_sq_dist(self, source_pos, target_pos):
        """
        Compute distances between each pair of nodes of the graph.

        Returns:
            (torch.tensor): the squared distances tensor.
        """
        num_edges = source_pos.shape[0]

        sq_dist = torch.zeros(num_edges)

        delta_pos = torch.cat(
            (
                source_pos[:, :2] - target_pos[:, :2],
                (((source_pos[:, 2] - target_pos[:, 2] - math.pi / 2) % math.pi) - math.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )

        for x3 in self.x3_axis:
            x3_mask = source_pos[:, 2] == x3

            delta_pos = torch.cat(
                (
                    source_pos[x3_mask, :2] - target_pos[x3_mask, :2],
                    (((source_pos[x3_mask, 2] - target_pos[x3_mask, 2] - math.pi / 2) % math.pi) - math.pi / 2).unsqueeze(1),
                ),
                dim=1,
            )

            sq_dist[x3_mask] = torch.bmm((delta_pos @ metric_tensor(x3, self.sigmas)).unsqueeze(1), delta_pos.unsqueeze(2)).flatten()

        return sq_dist

    def embed_on_graph(self, images, targets=None):
        """
        Embed images and targets on a the graph and return a list of Data object with the embedding.

        Args:
            images (torch.tensor): the images in format (N, C, H, W).
            targets (torch.tensor, optional): the target in format (N, D). Defaults to None.

        Raises:
            ValueError: in case images cannot be embedded on the graph due to dimension incompatibility.

        Returns:
            (list): the list of Data object with the embedding of images and targets on the graph.
        """
        num_images, channels, height, width = images.shape

        if height != self.nx2 or width != self.nx1:
            raise ValueError("Impossible to embed images on the graph, the dimensions don't fit")

        images = images.unsqueeze(1).permute(0, 1, 3, 4, 2)

        images = images.expand(num_images, self.nx3, self.nx2, self.nx1, channels)
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
        Access the Data object containing the graph

        Returns:
            (Data): the graph on a Data object.
        """
        return Data(edge_index=self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes)


def get_neighbors(graph_data, node_idx, return_weights=True):
    """
    Get the node indices of the neighbors of a given node.

    Args:
        graph_data (GraphData): the GraphData object containing information about the graph.
        node_idx ([type]): the node index of interest
        return_weights (bool, optional): the indicator to return weights associated to each neighbors.
            Defaults to True.

    Returns:
        (torch.tensor): the node indices of the neighbors of the node.
        (torch.tensor, optional): the weights of the edges between the node and its neighbors
    """
    mask = (graph_data.edge_index[0] == node_idx) & (graph_data.edge_weight > 0.0)
    neighbors = graph_data.edge_index[1, mask]

    if not return_weights:
        return neighbors

    weights = graph_data.edge_weight[mask]
    return neighbors, weights


def static_node_compression(node_index, kappa):
    """
    Randomly remove a given rate of nodes from the original tensor of node's indices

    Args:
        node_index (torch.tensor): the original node's indices.
        kappa (float): the rate of nodes to remove.

    Returns:
        (torch.tensor): the compressed node's indices.
    """
    num_nodes = node_index.nelement()

    num_to_remove = int(kappa * num_nodes)
    num_to_keep = num_nodes - num_to_remove

    mask = torch.tensor([False] * num_to_remove + [True] * num_to_keep)
    mask = shuffle(mask)

    return node_index[mask]


def static_edge_compression(edge_index, edge_weight, kappa):
    """
    Randomly remove a given rate of edges from the original tensor of edge's indices.

    Args:
        edge_index (torch.tensor): the original edge's indices.
        edge_weight (torch.tensor): the original edge's weights.
        kappa (float): the rate of edges to remove.

    Returns:
        (torch.tensor): the compressed edge's indices.
        (torch.tensor): the compressed edge's weights.
    """
    num_edges = edge_index.shape[1]

    num_to_remove = int(kappa * num_edges)
    num_to_keep = num_edges - num_to_remove

    mask = torch.tensor([False] * num_to_remove + [True] * num_to_keep)
    mask = shuffle(mask)

    return edge_index[:, mask], edge_weight[mask]
