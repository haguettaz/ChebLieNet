import math
import os
import sys
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from torch import FloatTensor, LongTensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor
from tqdm import tqdm

from ..liegroups.se2 import se2_matrix, se2_riemannian_sqdist
from ..liegroups.so3 import alphabetagamma2xyz, so3_matrix, so3_riemannian_sqdist, xyz2betagamma
from .gsp import get_fourier_basis, get_laplacian, get_norm_laplacian
from .utils import remove_duplicated_edges, to_undirected


class Graph:
    """
    Symbolic class representing a graph with nodes and edges. The main graph's operations are implemented
    in this class: Laplacian, eigen space and diffusion kernels.
    """

    def get_laplacian(self, norm=True, device: Optional[Device] = None):
        """
        Returns symmetric normalized graph laplacian .

        Args:
            norm (bool, optional): if True return normalized laplacian with eigenvalues in [-1, 1].
            device (Device, optional): computation device. Defaults to None.

        Returns:
            (SparseFloatTensor): laplacian.
        """
        if not hasattr(self, "laplacian"):
            if norm:
                self.laplacian = get_norm_laplacian(self.edge_index, self.edge_weight, self.num_nodes, 2.0, device)
            else:
                self.laplacian = get_laplacian(self.edge_index, self.edge_weight, self.num_nodes, device=device)

        return self.laplacian

    def get_eigen_space(self, norm=False) -> Tuple[ndarray, ndarray]:
        """
        Return graph eigen space, i.e. Laplacian eigen decomposition.

        Args:
            norm (bool, optional): if True, uses the normalized laplacian with eigenvalues in [-1, 1].

        Returns:
            (ndarray): Laplacian eigen values.
            (ndarray): Laplacian eigen vectors.
        """
        if not hasattr(self, "eigen_space"):
            self.eigen_space = get_fourier_basis(self.get_laplacian(norm))

        return self.eigen_space

    def diff_kernel(self, kernel: Callable) -> ndarray:
        """
        Return the diffusion kernel of the graph specified by the kernel imput.

        Args:
            tau (float): time constant.

        Returns:
            (ndarray): diffusion kernel.
        """
        lambdas, Phi = self.eigen_space
        return Phi @ np.diag(kernel(lambdas)) @ Phi.T

    @property
    def num_nodes(self) -> int:
        """
        Return the total number of nodes of the graph.

        Returns:
            (int): number of nodes.
        """
        return self.node_index.shape[0]

    @property
    def num_edges(self) -> int:
        """
        Return the total number of edges of the graph.

        Returns:
            (int): number of edges.
        """
        return self.edge_index.shape[1]

    def neighborhood(self, node_index: int) -> Tuple[LongTensor, FloatTensor, FloatTensor]:
        """
        Returns the node's neighborhood.

        Args:
            node_index (int): node index.

        Returns:
            (LongTensor): neighbors' indices.
            (FloatTensor): edges' weight from node to neighbors.
            (FloatTensor): squared riemannian distance from node to neighbors.
        """
        if not node_index in self.node_index:
            raise ValueError(f"{node_index} is not a valid index")

        mask = self.edge_index[0] == node_index
        return self.edge_index[1, mask], self.edge_weight[mask], self.edge_sqdist[mask]

    @property
    def is_connected(self):
        """
        Returns True is the graph is connected, that is it does not contain isolated node.

        Returns:
            (bool): True if the graph is connected.
        """
        return (self.node_index.repeat(1, self.num_edges) == self.edge_index[0]).sum(dim=1).min() > 0

    @property
    def is_undirected(self):
        """
        Returns True is the graph is undirected.

        Returns:
            (bool): True if the graph is undirected.
        """
        return torch.allclose(self.edge_index[0].sort(), self.edge_index[1].sort())

    def save(self, path_to_graph):
        """
        Save graph's attributes.
        """

        os.makedirs(path_to_graph, exist_ok=True)
        for attr in self.node_attributes:
            torch.save(getattr(self, attr), os.path.join(path_to_graph, f"{self.str_repr}_{attr}.pt"))
        for attr in self.edge_attributes:
            torch.save(getattr(self, attr), os.path.join(path_to_graph, f"{self.str_repr}_{attr}.pt"))

    def load(self, path_to_graph):
        """
        Load graph's attributes.
        """

        for attr in self.node_attributes:
            setattr(self, attr, torch.load(os.path.join(path_to_graph, f"{self.str_repr}_{attr}.pt")))
        for attr in self.edge_attributes:
            setattr(self, attr, torch.load(os.path.join(path_to_graph, f"{self.str_repr}_{attr}.pt")))

    def check_graph_exists(self, path_to_graph):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        for attr in self.node_attributes:
            if not os.path.exists(os.path.join(path_to_graph, f"{self.str_repr}_{attr}.pt")):
                return False

        for attr in self.edge_attributes:
            if not os.path.exists(os.path.join(path_to_graph, f"{self.str_repr}_{attr}.pt")):
                return False

        return True


class RandomSubGraph(Graph):
    """
    Symbolic class to generate a random sub-graph, that is a graph that is randomly created
    from another by edges' or nodes' sampling.

    Args:
        (Graph): parent class representing a graph.
    """

    def __init__(self, graph):
        """
        Initialization.

        Args:
            graph (Graph): original graph.
        """

        self.graph = graph
        self.lie_group = self.graph.lie_group

        for attr in graph.node_attributes:
            setattr(self, attr, getattr(graph, attr))

        for attr in graph.edge_attributes:
            setattr(self, attr, getattr(graph, attr))

        self.sub_node_index = self.node_index.clone()

    def edge_sampling(self, rate):
        """
        Randomly samples a given rate of edges from the original graph to generate a random sub-graph.

        Args:
            rate (float): rate of edges to sample.
        """
        # samples N (undirected) edges from the original graph based on their weights
        edge_attr = torch.stack((self.graph.edge_weight, self.graph.edge_sqdist))
        edge_index, edge_attr = remove_duplicated_edges(self.graph.edge_index, edge_attr)
        num_samples = math.ceil(rate * edge_attr[0].nelement())
        sampled_edges = torch.multinomial(edge_attr[0], num_samples)

        edge_index, edge_attr = to_undirected(
            edge_index[:, sampled_edges], edge_attr[:, sampled_edges], self_loop=False
        )

        self.edge_index = edge_index
        self.edge_weight = edge_attr[0]
        self.edge_sqdist = edge_attr[1]

    def node_sampling(self, rate):
        """
        Randomly samples a given rate of nodes from the original graph to generate a random sub-graph.

        Args:
            rate (float): rate of nodes to sample.
        """
        # samples N nodes from the original graph
        num_samples = math.floor(rate * self.graph.num_nodes)
        sampled_nodes, _ = torch.multinomial(torch.ones(self.graph.num_nodes), num_samples).sort()
        self.node_index = torch.arange(num_samples)
        self.sub_node_index = sampled_nodes.clone()

        for attr in self.graph.node_attributes:
            setattr(self, attr, getattr(self.graph, attr)[sampled_nodes])

        # selects edges between sampled nodes.
        node_mapping = torch.empty(self.graph.num_nodes, dtype=torch.long).fill_(-1)
        node_mapping[self.graph.node_index[sampled_nodes]] = self.node_index

        edge_index = node_mapping[self.graph.edge_index]
        mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
        self.edge_index = edge_index[:, mask]
        self.edge_weight = self.graph.edge_weight[mask]
        self.edge_sqdist = self.graph.edge_sqdist[mask]

    def cartesian_pos(self, axis=None) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """
        Returns the cartesian position of the graph's nodes.

        Args:
            axis (str, optional): position's axis. Defaults to None.

        Returns:
            (FloatTensor, optional): x position.
            (FloatTensor, optional): y position.
            (FloatTensor, optional): z position.
        """
        x, y, z = self.graph.cartesian_pos()

        if axis == "x":
            return x[self.sub_node_index]
        if axis == "y":
            return y[self.sub_node_index]
        if axis == "z":
            return z[self.sub_node_index]
        return x[self.sub_node_index], y[self.sub_node_index], z[self.sub_node_index]


class GEGraph(Graph):
    def __init__(self, uniform_sampling, sigmas, K, path_to_graph):
        """
        Initialization.

        Args:
            uniform_sampling (torch.tensor): [description]
            sigmas (tuple of floats): [description]
            K (int): [description]
        """
        super().__init__()

        if self.check_graph_exists(path_to_graph):
            print("Graph already exists: LOADING...")
            self.load(path_to_graph)
            print("Done!")

        else:
            print("Graph does not already exist: INITIALIZATION...")
            self._initnodes(uniform_sampling)
            self._initedges(sigmas, K)
            print("Done!")
            self.save(path_to_graph)
            print("Saved!")

    def _initnodes(self, uniform_sampling):
        _, V = uniform_sampling.shape

        for k, d in self.group_dim.items():
            setattr(self, f"node_{k}", uniform_sampling[d])

        self.node_index = torch.arange(V)

    def _initedges(self, sigmas, K):
        Gg = self.general_linear_group_element
        edge_sqdist = torch.empty(self.num_nodes * (K + 1))
        edge_index = torch.empty((2, self.num_nodes * (K + 1)), dtype=torch.long)
        Re = torch.diag(torch.tensor(sigmas))

        # compute all pairwise distances of the graph. WARNING: can possibly take a lot of time!!
        for idx in tqdm(self.node_index, file=sys.stdout):
            sqdist = self.riemannian_sqdist(Gg[idx], Gg, Re)
            values, indices = torch.topk(sqdist, largest=False, k=K + 1, sorted=False)
            edge_sqdist[idx * (K + 1) : (idx + 1) * (K + 1)] = values
            edge_index[0, idx * (K + 1) : (idx + 1) * (K + 1)] = idx
            edge_index[1, idx * (K + 1) : (idx + 1) * (K + 1)] = indices

        # remove duplicated edges and self-loops and make the graph undirected
        edge_index, edge_sqdist = remove_duplicated_edges(edge_index, edge_sqdist)
        self.edge_index, self.edge_sqdist = to_undirected(edge_index, edge_sqdist, self_loop=False)

        weight_kernel = lambda sqdistc, tc: torch.exp(-sqdistc / (4 * tc))
        self.edge_weight = weight_kernel(self.edge_sqdist, 0.2 * self.edge_sqdist.mean())


class SE2GEGraph(GEGraph):
    """
    Object representing a SE(2) group equivariant graph. It can be considered as a discretization of
    the SE(2) group where nodes corresponds to group elements and edges are proportional to the anisotropic
    SE(2) riemannian distances between group elements.
    """

    def __init__(self, uniform_sampling, sigmas, K, path_to_graph):
        """
        Initialization.

        Args:
            uniform_sampling (Tensor): uniform sampling of SE(2) group's elements.
            sigmas (tuple of floats): anisotropy's parameters.
            K (int): number of neighbors.
            path_to_graph (str): path to the folder to save graph.
        """

        self.str_repr = f"SE2GEGraph-V{uniform_sampling.size(1)}-E{uniform_sampling.size(1)*K}-Re{sigmas[0]}_{sigmas[1]}_{sigmas[2]}"
        self.lie_group = "se2"

        super().__init__(uniform_sampling, sigmas, K, path_to_graph)

    def riemannian_sqdist(self, Gg, Gh, Re):
        """
        Return the riemannian squared distance between GL(3) group elements Gg and Gh according to the
        riemannian metric Re.

        Args:
            Gg (Tensor): GL(3) group elements.
            Gh (Tensor): GL(3) group elements.
            Re (Tensor): riemannian metric.

        Returns:
            (Tensor): squared riemannian distance.
        """
        return se2_riemannian_sqdist(Gg, Gh, Re)

    @property
    def group_element(self):
        """
        Return the group elements of graph's vertices.

        Returns:
            (Tensor): SE(2) group's elements.
        """
        return self.node_x, self.node_y, self.node_theta

    @property
    def general_linear_group_element(self):
        """
        Return the general linear group elements of graph's vertices.

        Returns:
            (Tensor): GL(3) group's elements.
        """
        return se2_matrix(self.node_x, self.node_y, self.node_theta)

    @property
    def group_dim(self):
        return {"x": 0, "y": 1, "theta": 2}

    @property
    def size(self):
        # not compatible with node compression
        nsym = self.node_theta.unique().nelement()
        return (nsym, self.num_nodes // nsym)

    def cartesian_pos(self, axis=None):
        """
        Return the cartesian positions of the graph's vertices.

        Args:
            (str, optional): the cartesian axis to return, if None, return all axis. Defaults to None.

        Returns:
            (FloatTensor, optional): x positions.
            (FloatTensor, optional): y positions.
            (FloatTensor, optional): z positions.
        """
        if axis is None:
            return self.node_x, self.node_y, self.node_theta
        if axis == "x":
            return self.node_x
        if axis == "y":
            return self.node_y
        if axis == "z":
            return self.node_theta

    @property
    def node_attributes(self):
        return ("node_index", "node_x", "node_y", "node_theta")

    @property
    def edge_attributes(self):
        return ("edge_index", "edge_weight", "edge_sqdist")


class SO3GEGraph(GEGraph):
    """
    Object representing a SO(3) group equivariant graph. It can be considered as a discretization of
    the SO(3) group where nodes corresponds to group elements and edges are proportional to the anisotropic
    SO(3) riemannian distances between group elements.
    """

    def __init__(self, uniform_sampling, sigmas, K, path_to_graph):
        """
        Initialization.

        Args:
            uniform_sampling (Tensor): uniform sampling of SE(2) group's elements.
            sigmas (tuple of floats): anisotropy's parameters.
            K (int): number of neighbors.
            path_to_graph (str): path to the folder to save graph.
        """

        self.str_repr = f"SO3GEGraph-V{uniform_sampling.size(1)}-E{uniform_sampling.size(1)*K}-Re{sigmas[0]}_{sigmas[1]}_{sigmas[2]}"
        self.lie_group = "so3"

        super().__init__(uniform_sampling, sigmas, K, path_to_graph)

    def riemannian_sqdist(self, Gg, Gh, Re):
        """
        Return the riemannian squared distance between GL(3) group elements Gg and Gh according to the
        riemannian metric Re.

        Args:
            Gg (Tensor): GL(3) group elements.
            Gh (Tensor): GL(3) group elements.
            Re (Tensor): riemannian metric.

        Returns:
            (Tensor): squared riemannian distance.
        """
        return so3_riemannian_sqdist(Gg, Gh, Re)

    @property
    def group_element(self):
        """
        Return the group elements of graph's vertices.

        Returns:
            (Tensor): SO(3) group's elements.
        """
        return self.node_alpha, self.node_beta, self.node_gamma

    @property
    def general_linear_group_element(self):
        """
        Return the general linear group elements of graph's vertices.

        Returns:
            (Tensor): GL(3) group's elements.
        """
        return so3_matrix(self.node_alpha, self.node_beta, self.node_gamma)

    @property
    def group_dim(self):
        return {"alpha": 0, "beta": 1, "gamma": 2}

    @property
    def size(self):
        # not compatible with node compression
        nsym = self.node_alpha.unique().nelement()
        return (nsym, self.num_nodes // nsym)

    def cartesian_pos(self, axis=None):
        """
        Return the cartesian positions of the graph's vertices.

        Args:
            (str, optional): the cartesian axis to return. Defaults to None.

        Returns:
            (FloatTensor, optional): x positions.
            (FloatTensor, optional): y positions.
            (FloatTensor, optional): z positions.
        """
        x, y, z = alphabetagamma2xyz(self.node_alpha, self.node_beta, self.node_gamma)

        if axis == "x":
            return x
        if axis == "y":
            return y
        if axis == "z":
            return z

        return x, y, z

    @property
    def node_attributes(self):
        return ("node_index", "node_alpha", "node_beta", "node_gamma")

    @property
    def edge_attributes(self):
        return ("edge_index", "edge_weight", "edge_sqdist")
