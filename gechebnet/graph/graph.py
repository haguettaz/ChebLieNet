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

from ..liegroup.se2 import se2_anisotropic_square_riemannanian_distance, se2_log, se2_matrix
from ..liegroup.so3 import so3_anisotropic_square_riemannanian_distance, so3_log, so3_matrix
from ..liegroup.utils import alphabetagamma2xyz, xyz2alphabetagamma
from ..utils import rescale, sparse_tensor_to_sparse_array
from .optimization import repulsive_loss, repulsive_sampling
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

    def set_laplacian(self, norm=True, device: Optional[Device] = None):
        """
        Returns symmetric normalized graph laplacian

        Args:
            device (Device, optional): computation device. Defaults to None.

        Returns:
            (SparseFloatTensor): laplacian.
        """
        if norm:
            self.laplacian = get_norm_laplacian(self.edge_index, self.edge_weight, self.num_nodes, 2.0, device)
        else:
            self.laplacian = get_laplacian(self.edge_index, self.edge_weight, self.num_nodes, device=device)

    def set_sparse_laplacian(self, on: str, rate: float, norm=True, device: Optional[Device] = None):
        if on == "edges":
            edge_index, edge_weight = sparsify_on_edges(self.edge_index, self.edge_weight, rate)
        else:
            edge_index, edge_weight = sparsify_on_nodes(
                self.edge_index,
                self.edge_weight,
                self.node_index,
                self.num_nodes,
                rate,
            )

        if norm:
            self.laplacian = get_norm_laplacian(edge_index, edge_weight, self.num_nodes, 2.0, device)
        else:
            self.laplacian = get_laplacian(edge_index, edge_weight, self.num_nodes, device)

    @property
    def eigen_space(self) -> Tuple[ndarray, ndarray]:
        """
        Return graph eigen space, i.e. Laplacian eigen decomposition.

        Returns:
            (ndarray): Laplacian eigen values.
            (ndarray): Laplacian eigen vectors.
        """
        return get_fourier_basis(self.laplacian)

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
        knn: Optional[int] = 16,
        sigmas: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        weight_kernel: Optional[Callable] = None,
    ):
        """
        Inits a SE(2) group equivariant graph.
            1. Uniformly samples points on the SE(2) manifold.
            2. Init edges between nodes. Each node has at most knn neighbors, weight of edges are computed according to the
            Riemannian distance between them and the given weight kernel.
            3. Compress the graph according to the given compression algorithm.
            4. Init laplacian the symmetric normalized laplacian of the graph and store its maximum eigen value.

        Args:
            nx (int): x axis discretization.
            ny (int): y axis discretization.
            ntheta (int, optional): theta axis discretization. Defaults to 6.
            kappa (float, optional): edges compression rate. Defaults to 0.0.
            weight_kernel (callable, optional): weight kernel to use. Defaults to None.
            knn (int, optional): maximum number of connections of a vertex. Defaults to 16.
            sigmas (tuple, optional): anisotropy's parameters to compute anisotropic Riemannian distances. Defaults to (1., 1., 1.).
            device (Device): device. Defaults to None.
        """

        super().__init__()

        if weight_kernel is None:
            weight_kernel = lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac)

        self.nx, self.ny, self.ntheta = nx, ny, ntheta

        self._initnodes(nx * ny * ntheta)

        self._initedges(sigmas, knn if knn < self.num_nodes else self.num_nodes - 1, weight_kernel)

    def _initnodes(self, num_nodes: int):
        """
        Init node indices and positions (hypercube pose). The stored attributes are:
            - node_index (LongTensor): indices of nodes in format (num_nodes).
            - x (FloatTensor): x position of nodes in format (num_nodes) and in range (-inf, +inf).
            - y (FloatTensor): y position of nodes in format (num_nodes) and in range (-inf, +inf).
            - theta (FloatTensor): theta position of nodes in format (num_nodes) and in range [-pi/2, pi/2).

        Args:
            num_nodes (int): number of nodes to sample.
        """

        self.node_index = torch.arange(num_nodes)

        x_axis = torch.arange(0.0, self.nx)
        y_axis = torch.arange(0.0, self.ny)
        theta_axis = torch.arange(-math.pi / 2, math.pi / 2, math.pi / self.ntheta)

        theta, y, x = torch.meshgrid(theta_axis, y_axis, x_axis)

        self.node_x = x.flatten()
        self.node_y = y.flatten()
        self.node_theta = theta.flatten()

    def _initedges(
        self,
        sigmas: Tuple[float, float, float],
        knn: int,
        weight_kernel: Callable,
    ):
        """
        Init edge indices and attributes (weights). The stored attributes are:
            - edge_index (LongTensor): indices of edges in format (2, num_edges).
            - edge_weight (FloatTensor): weight of edges in format (num_edges).

        Args:
            sigmas (float,float,float): anisotropy's parameters to compute Riemannian distances.
            knn (int): maximum number of connections of a vertex.
            weight_kernel (callable): mapping from squared distance to weight value.
            kappa (float): edges' compression rate.
            device (Device): computation device.

        Raises:
            ValueError: kappa must be in [0, 1).
        """
        xi = Vi(torch.inverse(self.node_Gg()).reshape(self.num_nodes, -1))  # sources
        xj = Vj(self.node_Gg().reshape(self.num_nodes, -1))  # targets

        sqdist = se2_anisotropic_square_riemannanian_distance(
            xi,
            xj,
            sigmas,
        )
        edge_sqdist, neighbors = sqdist.Kmin_argKmin(knn + 1, dim=1)

        edge_index = torch.stack((self.node_index.repeat_interleave(knn + 1), neighbors.flatten()), dim=0)
        edge_sqdist = edge_sqdist.flatten()

        # remove duplicated edges and self-loops
        edge_index, edge_sqdist = remove_duplicated_edges(edge_index, edge_sqdist, self_loop=False)
        edge_index, edge_sqdist = to_undirected(edge_index, edge_sqdist)

        self.edge_index = edge_index
        self.edge_weight = weight_kernel(edge_sqdist, 1.09136 * edge_sqdist.mean())  # mean(sq_dist) -> weight = 0.4

    @property
    def nsym(self) -> int:
        """
        Returns the number of symmetry's layers.

        Returns:
            int: number of symmetry's layers.
        """
        return self.ntheta

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

    @property
    def centroid_index(self) -> int:
        """
        Returns the index of the centroid node of the graph.

        Returns:
            (int): centroid node's index.
        """

        mask = (
            self.node_x.isclose(self.node_x.median())
            & self.node_y.isclose(self.node_y.median())
            & self.node_theta.isclose(self.node_theta.median())
        )

        return self.node_index[mask]

    @property
    def lie_group(self):
        return "se2"


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
        nsamples: int,
        nalpha: Optional[int] = 6,
        knn: Optional[int] = 16,
        sigmas: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        weight_kernel: Optional[Callable] = None,
    ):
        """
        Inits a SO(3) group equivariant graph.
            1. Uniformly samples points on the SE(2) manifold.
            2. Init edges between nodes. Each node has at most knn neighbors, weight of edges are computed according to the
            Riemannian distance between them and the given weight kernel.
            3. Compress the graph according to the given compression algorithm.
            4. Init laplacian the symmetric normalized laplacian of the graph and store its maximum eigen value.

        Args:
            nsamples (int): number of samples on the pi-sphere
            nalpha (int, optional): alpha axis discretization. Defaults to 6.
            knn (int, optional): maximum number of connections of a vertex. Defaults to 16.
            sigmas (tuple, optional): anisotropy's parameters to compute anisotropic Riemannian distances. Defaults to (1., 1., 1.).
            weight_kernel (callable, optional): mapping from squared distance to weight value.
            kappa (float, optional): edges' compression rate. Defaults to 0.0.
            device (Device, optional): computation device. Defaults to None.
        """

        super().__init__()

        if weight_kernel is None:
            weight_kernel = lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac)

        self.nsamples = nsamples
        self.nalpha = nalpha  # alpha

        self._initnodes(nsamples * nalpha)
        self._initedges(sigmas, knn, weight_kernel)

    def _initnodes(self, num_nodes: int):
        """
        Init nodes on the SO(3) manifold. The stored attributes are:
            - node_index (LongTensor): indices of nodes in format (num_nodes).
            - alpha (FloatTensor): rotation about x axis in format (num_nodes) and in range [-pi/2, pi/2).
            - beta (FloatTensor): rotation about y axis in format (num_nodes) and in range [-pi, pi).
            - gamma (FloatTensor): rotation about z axis in format (num_nodes) and in range [-pi/2, pi/2).

        Args:
            num_nodes (int): number of nodes to sample.
            device (Device): computation device.
        """

        self.node_index = torch.arange(num_nodes)

        # uniform sampling on the sphere using a repulsive model
        x, y, z = repulsive_sampling(
            self.nsamples,
            loss_fn=lambda x_: repulsive_loss(x_, 1.0, 10.0),
            radius=math.pi,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            max_iter=25000,
        )

        # convert cartesian positions of the nodes on the sphere to beta and gamma rotations
        _, beta, gamma = xyz2alphabetagamma(x, y, z)

        alpha = torch.arange(-math.pi / 2, math.pi / 2, math.pi / self.nalpha)

        self.node_alpha = alpha.unsqueeze(1).expand(self.nalpha, self.nsamples).flatten()
        self.node_beta = beta.unsqueeze(0).expand(self.nalpha, self.nsamples).flatten()
        self.node_gamma = gamma.unsqueeze(0).expand(self.nalpha, self.nsamples).flatten()

    def _initedges(
        self,
        sigmas: Tuple[float, float, float],
        knn: int,
        weight_kernel: Callable,
    ):
        """
        Init edge indices and attributes (weights). The stored attributes are:
            - edge_index (LongTensor): indices of edges in format (2, num_edges).
            - edge_weight (FloatTensor): weight of edges in format (num_edges).

        Args:
            sigmas (tuple): anisotropy's parameters to compute anisotropic Riemannian distances.
            knn (int): maximum number of connections of a vertex.
            weight_kernel (callable): mapping from squared distance to weight value.
            kappa (float): edges' compression rate.
            device (Device): computation device.

        Raises:
            ValueError: kappa must be in [0, 1).
        """
        Gg = self.node_Gg().reshape(self.num_nodes, -1)
        Gh = self.node_Gg().inverse().reshape(self.num_nodes, -1)

        xi, xj, xi_t, xj_t = Vi(Gh), Vj(Gg), Vi(Gg), Vj(Gh)

        sqdist = so3_anisotropic_square_riemannanian_distance(xi, xj, xi_t, xj_t, sigmas)

        edge_sqdist, neighbors = sqdist.Kmin_argKmin(knn + 1, dim=1)

        edge_index = torch.stack((self.node_index.repeat_interleave(knn + 1), neighbors.flatten()), dim=0)
        edge_sqdist = edge_sqdist.flatten()

        # remove duplicated edges and self-loops
        edge_index, edge_sqdist = remove_duplicated_edges(edge_index, edge_sqdist, self_loop=False)
        edge_index, edge_sqdist = to_undirected(edge_index, edge_sqdist)

        self.edge_index = edge_index
        self.edge_weight = weight_kernel(edge_sqdist, 1.09136 * edge_sqdist.mean())  # mean(sq_dist) -> weight = 0.4

    @property
    def nsym(self) -> int:
        """
        Returns the number of symmetry's layers.

        Returns:
            int: number of symmetry's layers.
        """
        return self.nalpha

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

    @property
    def centroid_index(self) -> int:
        """
        Returns the index of the centroid node of the graph.

        Returns:
            (int): centroid node's index.
        """
        return 0

    @property
    def lie_group(self):
        return "so3"
