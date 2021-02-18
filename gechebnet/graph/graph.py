import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pykeops.torch import Vi, Vj
from torch import FloatTensor, LongTensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor

from ..liegroup.se2 import se2_matrix, se2_riemannian_sqdist
from ..liegroup.so3 import alphabetagamma2xyz, so3_matrix, so3_riemannian_sqdist, xyz2betagamma
from ..utils import rescale, sparse_tensor_to_sparse_array
from .optimization import repulsive_loss, repulsive_sampling
from .polyhedron import polyhedron_division, polyhedron_init
from .signal_processing import get_fourier_basis, get_laplacian, get_norm_laplacian
from .sparsification import sparsify_on_edges, sparsify_on_nodes
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

    def get_laplacian(self, norm=True, device: Optional[Device] = None):
        """
        Returns symmetric normalized graph laplacian

        Args:
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
        if not hasattr(self, "node_proj"):
            return signal

        return signal[..., self.node_proj]

    @property
    def eigen_space(self, norm=False) -> Tuple[ndarray, ndarray]:
        """
        Return graph eigen space, i.e. Laplacian eigen decomposition.

        Returns:
            (ndarray): Laplacian eigen values.
            (ndarray): Laplacian eigen vectors.
        """
        return get_fourier_basis(self.get_laplacian(norm))

    def diff_kernel(self, kernel: Callable) -> ndarray:
        """
        Return the diffusion kernel of the graph specified by the kernel imput.

        Args:
            tau (float): time constant.

        Returns:
            ndarray: diffusion kernel.
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

    def neighbors_weights(self, node_index):
        mask = self.edge_index[0] == node_index
        neighbors_index = self.edge_index[1, mask]
        weights = torch.zeros(self.num_nodes)
        weights[neighbors_index] = self.edge_weight[mask]
        return weights

    def neighbors_sqdists(self, node_index):
        mask = self.edge_index[0] == node_index
        neighbors_index = self.edge_index[1, mask]
        sqdists = torch.empty(self.num_nodes).fill_(math.nan)
        sqdists[neighbors_index] = self.edge_sqdist[mask]
        return sqdists

    def dirac(self, node_idx: int = 0, lib: str = "numpy") -> Union[ndarray, FloatTensor]:
        """
        Return a dirac function centered on a given node index.

        Args:
            node_idx (int, optional): node index. Defaults to 0.
            lib (str, optional): used library. Defaults to "numpy".

        Raises:
            ValueError: lib must be 'numpy' or 'pytorch'

        Returns:
            Union[ndarray, FloatTensor]: dirac ndarray or tensor.
        """
        if lib not in {"numpy", "pytorch"}:
            raise ValueError(f"{lib} is not a valid value for lib: must be 'numpy' or 'pytorch'")

        if lib == "numpy":
            f = np.zeros(self.num_nodes)
        else:
            f = torch.zeros(self.num_nodes)

        f[node_idx] = 1.0
        return f

    # can possibily crash for graph with too high number of vertices and edges
    @property
    def contains_isolated_node(self):
        return (self.node_index.repeat(1, self.num_edges) == self.edge_index[0]).sum(dim=1).min() < 1


class RandomSubGraph(Graph):
    def __init__(self, graph):

        self.graph = graph
        self.lie_group = self.graph.lie_group
        self.nx1, self.nx2, self.nx3 = self.graph.nx1, self.graph.nx2, self.graph.nx3

        self._initnodes(graph)
        self._initedges(graph)

    def _initnodes(self, graph):
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
        self.edge_index = self.graph.edge_index.clone()
        self.edge_weight = self.graph.edge_weight.clone()
        self.edge_sqdist = self.graph.edge_sqdist.clone()

    def edge_sampling(self, rate):
        # samples N (undirected) edges to keep
        edge_attr = torch.stack((self.graph.edge_weight, self.graph.edge_sqdist))
        edge_index, edge_attr = remove_duplicated_edges(self.graph.edge_index, edge_attr, self_loop=False)
        num_samples = math.ceil(rate * edge_attr[0].nelement())  # num edges to keep
        sampled_edges = torch.multinomial(edge_attr[0], num_samples)  # edge_attr[0] corresponds to weights

        edge_index, edge_attr = to_undirected(edge_index[:, sampled_edges], edge_attr[:, sampled_edges])

        self.edge_index = edge_index
        self.edge_weight = edge_attr[0]
        self.edge_sqdist = edge_attr[1]

    def node_sampling(self, rate):
        # samples N nodes to keep
        num_samples = math.floor(rate * self.graph.num_nodes)  # num nodes to keep
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

        # keeps edges between sampled nodes
        node_mapping = torch.empty(self.graph.num_nodes, dtype=torch.long).fill_(-1)
        node_mapping[self.graph.node_index[sampled_nodes]] = self.node_index

        edge_index = node_mapping[self.graph.edge_index]
        mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
        self.edge_index = edge_index[:, mask]
        self.edge_weight = self.graph.edge_weight[mask]
        self.edge_sqdist = self.graph.edge_sqdist[mask]

    def node_pos(self, axis=None) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
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
            2. Init edges between nodes. Each node has at most K neighbors, weight of edges are computed according to the
            Riemannian distance between them and the given weight kernel.
            3. Compress the graph according to the given compression algorithm.
            4. Init laplacian the symmetric normalized laplacian of the graph and store its maximum eigen value.

        Args:
            nx (int): x axis discretization.
            ny (int): y axis discretization.
            ntheta (int, optional): theta axis discretization. Defaults to 6.
            kappa (float, optional): edges compression rate. Defaults to 0.0.
            weight_kernel (callable, optional): weight kernel to use. Defaults to None.
            K (int, optional): maximum number of connections of a vertex. Defaults to 16.
            sigmas (tuple, optional): anisotropy's parameters to compute anisotropic Riemannian distances. Defaults to (1., 1., 1.).
            device (Device): device. Defaults to None.
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
        Init node indices and positions (hypercube pose). The stored attributes are:
            - node_index (LongTensor): indices of nodes in format (num_nodes).
            - x (FloatTensor): x position of nodes in format (num_nodes) and in range (-inf, +inf).
            - y (FloatTensor): y position of nodes in format (num_nodes) and in range (-inf, +inf).
            - theta (FloatTensor): theta position of nodes in format (num_nodes) and in range [-pi/2, pi/2).

        Args:
            num_nodes (int): number of nodes to sample.
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
        Init edge indices and attributes (weights). The stored attributes are:
            - edge_index (LongTensor): indices of edges in format (2, num_edges).
            - edge_weight (FloatTensor): weight of edges in format (num_edges).

        Args:
            sigmas (float,float,float): anisotropy's parameters to compute Riemannian distances.
            K (int): maximum number of connections of a vertex.
            weight_kernel (callable): mapping from squared distance to weight value.
            device (Device): computation device.

        Raises:
            ValueError: kappa must be in [0, 1).
        """

        # xi = Vi(torch.inverse(self.node_Gg()).reshape(self.num_nodes, -1))  # sources
        # xj = Vj(self.node_Gg().reshape(self.num_nodes, -1))  # targets

        # sqdist = se2_anisotropic_square_riemannanian_distance(
        #     xi,
        #     xj,
        #     sigmas,
        # )
        # edge_sqdist, neighbors = sqdist.Kmin_argKmin(K + 1, dim=1)

        # edge_index = torch.stack((self.node_index.repeat_interleave(K + 1), neighbors.flatten()), dim=0)
        # edge_sqdist = edge_sqdist.flatten()

        Gg = self.node_Gg()
        edge_sqdist = torch.empty(self.num_nodes * (K + 1))
        edge_index = torch.empty((2, self.num_nodes * (K + 1)), dtype=torch.long)
        Re = torch.diag(torch.tensor(sigmas))

        for idx in self.node_index:
            sqdist = se2_riemannian_sqdist(Gg[idx], Gg, Re)
            values, indices = torch.topk(sqdist, largest=False, k=K + 1, sorted=False)
            edge_sqdist[idx * (K + 1) : (idx + 1) * (K + 1)] = values
            edge_index[0, idx * (K + 1) : (idx + 1) * (K + 1)] = idx
            edge_index[1, idx * (K + 1) : (idx + 1) * (K + 1)] = indices

        # remove duplicated edges and self-loops
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

        Returns:
            (FloatTensor): x nodes' positions.
            (FloatTensor): y nodes' positions.
            (FloatTensor): z nodes' positions.
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
    Object representing a SO(3) group equivariant graph. It can be considered as a discretization of
    the SO(3) group where nodes corresponds to group elements and edges are proportional to the anisotropic
    Riemannian distances between group elements.

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
            1. Uniformly samples points on the SE(2) manifold.
            2. Init edges between nodes. Each node has at most K neighbors, weight of edges are computed according to the
            Riemannian distance between them and the given weight kernel.
            3. Compress the graph according to the given compression algorithm.
            4. Init laplacian the symmetric normalized laplacian of the graph and store its maximum eigen value.

        Args:
            nsamples (int): number of samples on the pi-sphere
            nalpha (int, optional): alpha axis discretization. Defaults to 6.
            K (int, optional): maximum number of connections of a vertex. Defaults to 16.
            sigmas (tuple, optional): anisotropy's parameters to compute anisotropic Riemannian distances. Defaults to (1., 1., 1.).
            weight_kernel (callable, optional): mapping from squared distance to weight value.
            kappa (float, optional): edges' compression rate. Defaults to 0.0.
            device (Device, optional): computation device. Defaults to None.
        """

        super().__init__()

        self.nx3 = nalpha
        self.lie_group = "so3"
        if weight_kernel is None:
            weight_kernel = lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac)

        # self.nsamples = nsamples
        # self.nalpha = nalpha
        self._initnodes(polyhedron, level, nalpha)
        # self._initnodes(nsamples * nalpha)
        self._initedges(sigmas, K if K < self.num_nodes else self.num_nodes - 1, weight_kernel)

    def _initnodes(self, polyhedron, level, nalpha):
        # uniformly samples on the sphere by polyhedron subdivisions -> uniformly sampled beta and gamma
        vertices, faces = polyhedron_init(polyhedron)
        x, y, z = polyhedron_division(vertices, faces, level)
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

    # def _initnodes(self, num_nodes: int):
    #     """
    #     Init nodes on the SO(3) manifold. The stored attributes are:
    #         - node_index (LongTensor): indices of nodes in format (num_nodes).
    #         - alpha (FloatTensor): rotation about x axis in format (num_nodes) and in range [-pi/2, pi/2).
    #         - beta (FloatTensor): rotation about y axis in format (num_nodes) and in range [-pi, pi).
    #         - gamma (FloatTensor): rotation about z axis in format (num_nodes) and in range [-pi/2, pi/2).

    #     Args:
    #         num_nodes (int): number of nodes to sample.
    #         device (Device): computation device.
    #     """

    #     self.node_index = torch.arange(num_nodes)

    #     # uniform sampling on the sphere using a repulsive model
    #     x, y, z = repulsive_sampling(
    #         self.nsamples,
    #         loss_fn=lambda x_: repulsive_loss(x_, 1.0, 10.0),
    #         radius=math.pi,
    #         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #         max_iter=25000,
    #     )

    #     # convert cartesian positions of the nodes on the sphere to beta and gamma rotations
    #     _, beta, gamma = xyz2alphabetagamma(x, y, z)

    #     alpha = torch.arange(-math.pi / 2, math.pi / 2, math.pi / self.nalpha)

    #     self.node_alpha = alpha.unsqueeze(1).expand(self.nalpha, self.nsamples).flatten()
    #     self.node_beta = beta.unsqueeze(0).expand(self.nalpha, self.nsamples).flatten()
    #     self.node_gamma = gamma.unsqueeze(0).expand(self.nalpha, self.nsamples).flatten()

    def _initedges(
        self,
        sigmas: Tuple[float, float, float],
        K: int,
        weight_kernel: Callable,
    ):
        """
        Init edge indices and attributes (weights). The stored attributes are:
            - edge_index (LongTensor): indices of edges in format (2, num_edges).
            - edge_weight (FloatTensor): weight of edges in format (num_edges).

        Args:
            sigmas (tuple): anisotropy's parameters to compute anisotropic Riemannian distances.
            K (int): maximum number of connections of a vertex.
            weight_kernel (callable): mapping from squared distance to weight value.
            kappa (float): edges' compression rate.
            device (Device): computation device.

        Raises:
            ValueError: kappa must be in [0, 1).
        """
        # Gg = self.node_Gg().reshape(self.num_nodes, -1)
        # Gh = self.node_Gg().inverse().reshape(self.num_nodes, -1)

        # xi, xj = Vi(Gh), Vj(Gg)

        # sqdist = so3_anisotropic_square_riemannanian_distance(xi, xj, sigmas)

        # edge_sqdist, neighbors = sqdist.Kmin_argKmin(k + 1, dim=1)

        # edge_index = torch.stack((self.node_index.repeat_interleave(K + 1), neighbors.flatten()), dim=0)
        # edge_sqdist = edge_sqdist.flatten()

        Gg = self.node_Gg()
        edge_sqdist = torch.empty(self.num_nodes * (K + 1))
        edge_index = torch.empty((2, self.num_nodes * (K + 1)), dtype=torch.long)
        Re = torch.diag(torch.tensor(sigmas))

        for idx in self.node_index:
            sqdist = so3_riemannian_sqdist(Gg[idx], Gg, Re)
            values, indices = torch.topk(sqdist, largest=False, k=K + 1, sorted=False)
            edge_sqdist[idx * (K + 1) : (idx + 1) * (K + 1)] = values
            edge_index[0, idx * (K + 1) : (idx + 1) * (K + 1)] = idx
            edge_index[1, idx * (K + 1) : (idx + 1) * (K + 1)] = indices

        # remove duplicated edges and self-loops
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
            (FloatTensor, optional): x nodes' positions.
            (FloatTensor, optional): y nodes' positions.
            (FloatTensor, optional): z nodes' positions.
        """
        return alphabetagamma2xyz(self.node_alpha, self.node_beta, self.node_gamma, axis)
