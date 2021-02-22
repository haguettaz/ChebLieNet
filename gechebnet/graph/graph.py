import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy import ndarray

# from pykeops.torch import Vi, Vj
from torch import FloatTensor, LongTensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor

from ..liegroup.se2 import se2_matrix, se2_riemannian_sqdist
from ..liegroup.so3 import alphabetagamma2xyz, so3_matrix, so3_riemannian_sqdist, xyz2betagamma
from ..utils import rescale, sparse_tensor_to_sparse_array
from .optimization import repulsive_loss, repulsive_sampling
from .polyhedron import SphericalPolyhedron
from .signal_processing import get_fourier_basis, get_laplacian, get_norm_laplacian
from .utils import remove_duplicated_edges, to_undirected


class Graph:
    """
    Symbolic class representing a graph with nodes and edges. The main graph's operations are implemented
    in this class: Laplacian, eigen space and diffusion kernels.
    """

    def __init__(self, *arg, **kwargs):
        """
        Init the graph attributes with empty tensors
        """
        self.node_index = LongTensor()

        self.edge_index = LongTensor()
        self.edge_weight = FloatTensor()
        self.edge_sqdist = FloatTensor()

    def get_laplacian(self, norm=True, device: Optional[Device] = None) -> SparseFloatTensor:
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

    def project(self, signal):
        """
        Projects a signal on the graph.

        Args:
            signal (Tensor): signal to project.

        Returns:
            (Tensor): projected signal.
        """
        if not hasattr(self, "node_proj"):
            return signal

        return signal[..., self.node_proj]

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
        mask = self.edge_index[0] == node_index
        return self.edge_index[1, mask], self.edge_weight[mask], self.edge_sqdist[mask]

    @property
    def is_connected():
        """
        Returns True is the graph is connected, that is it does not contain isolated node.

        Returns:
            (bool): True if the graph is connected.
        """
        return (self.node_index.repeat(1, self.num_edges) == self.edge_index[0]).sum(dim=1).min() > 0

    @property
    def is_undirected():
        """
        Returns True is the graph is undirected.

        Returns:
            (bool): True if the graph is undirected.
        """
        return torch.allclose(self.edge_index[0].sort(), self.edge_index[1].sort())


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
        self.nx1, self.nx2, self.nx3 = self.graph.nx1, self.graph.nx2, self.graph.nx3

        self._initnodes(graph)
        self._initedges(graph)

    def _initnodes(self, graph):
        """
        Inits nodes of the sub-graph.

        Args:
            graph (Graph): original graph.
        """
        self.node_index = self.graph.node_index

        if self.lie_group == "se2":
            self.node_x = self.graph.node_x.clone()
            self.node_y = self.graph.node_y.clone()
            self.node_theta = self.graph.node_theta.clone()

        elif self.lie_group == "so3":
            self.node_alpha = self.graph.node_alpha.clone()
            self.node_beta = self.graph.node_beta.clone()
            self.node_gamma = self.graph.node_gamma.clone()

    def _initedges(self, graph):
        """
        Inits edges of the sub-graph.

        Args:
            graph (Graph): original graph.
        """
        self.edge_index = self.graph.edge_index.clone()
        self.edge_weight = self.graph.edge_weight.clone()
        self.edge_sqdist = self.graph.edge_sqdist.clone()

    def edge_sampling(self, rate):
        """
        Randomly samples a given rate of edges from the original graph to generate a random sub-graph.

        Args:
            rate (float): rate of edges to sample.
        """
        # samples N (undirected) edges from the original graph based on their weights
        edge_attr = torch.stack((self.graph.edge_weight, self.graph.edge_sqdist))
        edge_index, edge_attr = remove_duplicated_edges(self.graph.edge_index, edge_attr, self_loop=False)
        num_samples = math.ceil(rate * edge_attr[0].nelement())
        sampled_edges = torch.multinomial(edge_attr[0], num_samples)

        edge_index, edge_attr = to_undirected(edge_index[:, sampled_edges], edge_attr[:, sampled_edges])

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
        self.node_proj = sampled_nodes.clone()

        # sets group attributes of sampled nodes
        if self.lie_group == "se2":
            self.node_x = self.graph.node_x[sampled_nodes]
            self.node_y = self.graph.node_y[sampled_nodes]
            self.node_theta = self.graph.node_theta[sampled_nodes]

        elif self.lie_group == "so3":
            self.node_alpha = self.graph.node_alpha[sampled_nodes]
            self.node_beta = self.graph.node_beta[sampled_nodes]
            self.node_gamma = self.graph.node_gamma[sampled_nodes]

        # selects edges between sampled nodes.
        node_mapping = torch.empty(self.graph.num_nodes, dtype=torch.long).fill_(-1)
        node_mapping[self.graph.node_index[sampled_nodes]] = self.node_index

        edge_index = node_mapping[self.graph.edge_index]
        mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
        self.edge_index = edge_index[:, mask]
        self.edge_weight = self.graph.edge_weight[mask]
        self.edge_sqdist = self.graph.edge_sqdist[mask]

    def node_pos(self, axis=None) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """
        Returns the cartesian position of the graph's nodes.

        Args:
            axis (str, optional): position's axis. Defaults to None.

        Returns:
            (FloatTensor, optional): x position.
            (FloatTensor, optional): y position.
            (FloatTensor, optional): z position.
        """
        if self.lie_group == "se2":
            if axis is None:
                return self.node_x, self.node_y, self.node_theta
            if axis == "x":
                return self.node_x
            if axis == "y":
                return self.node_y
            if axis == "z":
                return self.node_theta

        elif self.lie_group == "so3":
            return alphabetagamma2xyz(self.node_alpha, self.node_beta, self.node_gamma, axis)


class SE2GEGraph(Graph):
    """
    Object representing a SE(2) group equivariant graph. It can be considered as a discretization of
    the SE(2) group where nodes corresponds to group elements and edges are proportional to the anisotropic
    Riemannian distances between group elements.

    Args:
        (Graph): parent class representing a graph.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        ntheta: Optional[int] = 6,
        K: Optional[int] = 16,
        sigmas: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        weight_kernel: Optional[Callable] = None,
    ):
        """
        Inits a SE(2) group equivariant graph.
            1. Uniformly samples points on the SE(2) manifold.
            2. Init edges between nodes. Each node has at most K neighbors, edges' weights depend on the riemannian distances between nodes.

        Args:
            nx (int): discretization of the x axis.
            ny (int): discretization of the y axis.
            ntheta (int, optional): discretization of the theta axis. Defaults to 6.
            K (int, optional): maximum number of connections of a vertex. Defaults to 16.
            sigmas (tuple, optional): anisotropy's parameters to compute anisotropic Riemannian distances. Defaults to (1., 1., 1.).
            weight_kernel (callable, optional): weight kernel to use. Defaults to None.
        """

        super().__init__()

        self.nx1, self.nx2, self.nx3 = nx, ny, ntheta
        self.lie_group = "se2"

        if weight_kernel is None:
            weight_kernel = lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac)

        self._initnodes(nx, ny, ntheta)
        self._initedges(sigmas, K if K < self.num_nodes else self.num_nodes - 1, weight_kernel)

    def _initnodes(self, nx, ny, ntheta):
        """
        Init nodes' index and associated group elements. The stored attributes are:
            - node_index (LongTensor): indices of nodes in format (num_nodes).
            - node_x (FloatTensor): x attributes of the associated group elements in format (num_nodes) and in range [0, 1).
            - node_y (FloatTensor): y attributes of the associated group elements in format (num_nodes) and in range [0, 1).
            - node_theta (FloatTensor): theta attributes of the associated group elements in format (num_nodes) and in range [-pi/2, pi/2).

        Args:
            nx (int): discretization of the x axis.
            ny (int): discretization of the y axis.
            ntheta (int): discretization of the theta axis.
        """

        self.node_index = torch.arange(nx * ny * ntheta)

        if nx == 1:
            x_axis = torch.zeros(1)
        else:
            x_axis = torch.arange(0.0, 1.0, 1 / nx)

        if ny == 1:
            y_axis = torch.zeros(1)
        else:
            y_axis = torch.arange(0.0, 1.0, 1 / ny)

        if ntheta == 1:
            theta_axis = torch.zeros(1)
        else:
            theta_axis = torch.arange(-math.pi / 2, math.pi / 2, math.pi / ntheta)

        theta, y, x = torch.meshgrid(theta_axis, y_axis, x_axis)

        self.node_x = x.flatten()
        self.node_y = y.flatten()
        self.node_theta = theta.flatten()

    def _initedges(
        self,
        sigmas: Tuple[float, float, float],
        K: int,
        weight_kernel: Callable,
    ):
        """
        Inits edges' indices and attributes (weights and squared riemannian distances). The stored attributes are:
            - edge_index (LongTensor): indices of edges in format (2, num_edges).
            - edge_weight (FloatTensor): weight of edges in format (num_edges).
            - edge_sqdist (FloatTensor): squared riemannian distances between connected nodes in format (num_edges).

        Args:
            sigmas (float,float,float): anisotropy's parameters to compute riemannian distances.
            K (int): maximum number of connections of a vertex.
            weight_kernel (callable): mapping from squared riemannian distance to weight.
        """

        Gg = self.node_Gg()
        edge_sqdist = torch.empty(self.num_nodes * (K + 1))
        edge_index = torch.empty((2, self.num_nodes * (K + 1)), dtype=torch.long)
        Re = torch.diag(torch.tensor(sigmas))

        # compute all pairwise distances of the graph. WARNING: can possibly take a lot of time!!
        for idx in self.node_index:
            sqdist = se2_riemannian_sqdist(Gg[idx], Gg, Re)
            values, indices = torch.topk(sqdist, largest=False, k=K + 1, sorted=False)
            edge_sqdist[idx * (K + 1) : (idx + 1) * (K + 1)] = values
            edge_index[0, idx * (K + 1) : (idx + 1) * (K + 1)] = idx
            edge_index[1, idx * (K + 1) : (idx + 1) * (K + 1)] = indices

        # remove duplicated edges and self-loops and make the graph undirected
        edge_index, edge_sqdist = remove_duplicated_edges(edge_index, edge_sqdist, self_loop=False)
        edge_index, edge_sqdist = to_undirected(edge_index, edge_sqdist)

        self.edge_index = edge_index
        self.edge_sqdist = edge_sqdist
        self.edge_weight = weight_kernel(edge_sqdist, 0.2 * edge_sqdist.mean())

    def node_Gg(self, device=None) -> FloatTensor:
        """
        Returns the matrix formulation of group elements.

        Args:
            device (Device): computation device.

        Returns:
            (FloatTensor): nodes' in matrix formulation
        """
        return se2_matrix(self.node_x, self.node_y, self.node_theta, device=device)

    def node_pos(self, axis=None) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """
        Return the cartesian positions of the nodes of the graph.

        Args:
            axis (str, optional): position's axis. Defaults to None.

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


class SO3GEGraph(Graph):
    """
    Object representing a SO(3) group equivariant graph. It can be considered as a discretization of the SO(3) group where nodes
    corresponds to group elements and edges are proportional to the anisotropic Riemannian distances between group elements.

    Args:
        (Graph): parent class representing a graph.
    """

    def __init__(
        self,
        polyhedron: str,
        level: int,
        nalpha: Optional[int] = 6,
        K: Optional[int] = 16,
        sigmas: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        weight_kernel: Optional[Callable] = None,
    ):
        """
        Inits a SO(3) group equivariant graph.
            1. Uniformly samples points on the SO(3) manifold.
            2. Init edges between nodes. Each node has at most K neighbors and edges' weights depend on
            riemannian distances between nodes.

        Args:
            polyhedron (str): polyhedron of the polyhedral method to uniformly sample on S2.
            level (int): level of the polyhedral method.
            nalpha (int, optional): discretization of the alpha axis. Defaults to 6.
            K (int, optional): maximum number of connections of a vertex. Defaults to 16.
            sigmas (tuple, optional): anisotropy's parameters to compute anisotropic Riemannian distances. Defaults to (1., 1., 1.).
            weight_kernel (callable, optional): mapping from squared distance to weight value.
        """

        super().__init__()

        self.nx3 = nalpha
        self.lie_group = "so3"
        if weight_kernel is None:
            weight_kernel = lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac)

        self._initnodes(polyhedron, level, nalpha)
        self._initedges(sigmas, K if K < self.num_nodes else self.num_nodes - 1, weight_kernel)

    def _initnodes(self, polyhedron: str, level: int, nalpha: int):
        """
        Init nodes' index and associated group elements. The stored attributes are:
            - node_index (LongTensor): indices of nodes in format (num_nodes).
            - node_alpha (FloatTensor): alpha attributes of the associated group elements in format (num_nodes) and in range [-pi/2, pi/2).
            - node_beta (FloatTensor): beta attributes of the associated group elements in format (num_nodes) and in range [-pi, pi).
            - node_gamma (FloatTensor): gamma attributes of the associated group elements in format (num_nodes) and in range [-pi/2, pi/2).

        Args:
            polyhedron (str): polyhedron of the polyhedral method to uniformly sample on S2.
            level (int): level of the polyhedral method.
            nalpha (int): discretization of the alpha axis.
        """
        # uniformly samples on the sphere by polyhedron subdivisions -> uniformly sampled beta and gamma
        spherical_polyhedron = SphericalPolyhedron(polyhedron)
        x, y, z = spherical_polyhedron.spherical_sampling(level)
        beta, gamma = xyz2betagamma(x, y, z)

        # uniformly samples alpha
        if nalpha == 1:
            alpha = torch.zeros(1)
        else:
            alpha = torch.arange(-math.pi / 2, math.pi / 2, math.pi / nalpha)

        # add nodes' positions attributes
        self.node_alpha = alpha.unsqueeze(1).expand(nalpha, beta.shape[0]).flatten()
        self.node_beta = beta.unsqueeze(0).expand(nalpha, beta.shape[0]).flatten()
        self.node_gamma = gamma.unsqueeze(0).expand(nalpha, beta.shape[0]).flatten()

        self.node_index = torch.arange(self.node_alpha.shape[0])

    def _initedges(
        self,
        sigmas: Tuple[float, float, float],
        K: int,
        weight_kernel: Callable,
    ):
        """
        Inits edges' indices and attributes (weights and squared riemannian distances). The stored attributes are:
            - edge_index (LongTensor): indices of edges in format (2, num_edges).
            - edge_weight (FloatTensor): weight of edges in format (num_edges).
            - edge_sqdist (FloatTensor): squared riemannian distances between connected nodes in format (num_edges).

        Args:
            sigmas (float,float,float): anisotropy's parameters to compute riemannian distances.
            K (int): maximum number of connections of a vertex.
            weight_kernel (callable): mapping from squared riemannian distance to weight.
        """

        Gg = self.node_Gg()
        edge_sqdist = torch.empty(self.num_nodes * (K + 1))
        edge_index = torch.empty((2, self.num_nodes * (K + 1)), dtype=torch.long)
        Re = torch.diag(torch.tensor(sigmas))

        # compute all pairwise distances of the graph. WARNING: can possibly take a lot of time!!
        for idx in self.node_index:
            sqdist = so3_riemannian_sqdist(Gg[idx], Gg, Re)
            values, indices = torch.topk(sqdist, largest=False, k=K + 1, sorted=False)
            edge_sqdist[idx * (K + 1) : (idx + 1) * (K + 1)] = values
            edge_index[0, idx * (K + 1) : (idx + 1) * (K + 1)] = idx
            edge_index[1, idx * (K + 1) : (idx + 1) * (K + 1)] = indices

        # remove duplicated edges and self-loops and make the graph undirected
        edge_index, edge_sqdist = remove_duplicated_edges(edge_index, edge_sqdist, self_loop=False)
        edge_index, edge_sqdist = to_undirected(edge_index, edge_sqdist)

        self.edge_index = edge_index
        self.edge_sqdist = edge_sqdist
        self.edge_weight = weight_kernel(edge_sqdist, 0.2 * edge_sqdist.mean())  # mean(sq_dist) -> weight = 0.4

    def node_Gg(self, device=None) -> FloatTensor:
        """
        Returns the matrix formulation of group elements.

        Args:
            device (Device): computation device.

        Returns:
            (FloatTensor): nodes' in matrix formulation
        """
        return so3_matrix(self.node_alpha, self.node_beta, self.node_gamma, device=device)

    def node_pos(self, axis=None):
        """
        Returns the cartesian positions of the nodes of the graph.

        Returns:
            (FloatTensor, optional): x positions.
            (FloatTensor, optional): y positions.
            (FloatTensor, optional): z positions.
        """
        return alphabetagamma2xyz(self.node_alpha, self.node_beta, self.node_gamma, axis)
