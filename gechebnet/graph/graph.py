import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
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
from .compression import multinomial_compression
from .optimization import repulsive_loss, repulsive_sampling
from .signal_processing import get_fourier_basis, get_laplacian
from .utils import remove_directed_edges, remove_duplicated_edges, remove_self_loops


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

    def neighborhood(self, node_idx: int) -> Tuple[LongTensor, FloatTensor]:
        """
        Return neighborhood of a given node.

        Args:
            node_idx (int): node index.

        Returns:
            (LongTensor): neighbours index.
            (FloatTensor): neighbours weight.
        """
        mask = self.edge_index[0] == node_idx
        neighbors = self.edge_index[1, mask]

        weights = self.edge_weight[mask]
        return neighbors, weights

    def laplacian(self, device: Optional[Device] = None):
        """
        Returns symmetric normalized graph laplacian

        Args:
            device (Device, optional): computation device. Defaults to None.

        Returns:
            (SparseFloatTensor): laplacian.
        """
        return get_laplacian(self.edge_index, self.edge_weight, self.num_nodes, device=device)

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
        kappa: Optional[float] = 0.0,
        device: Device = None,
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

        self._initnodes(nsamples * nalpha, device)
        self._initedges(sigmas, knn, weight_kernel, kappa, device)
        self._initprojection()

    def _initnodes(self, num_nodes: int, device: Device):
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

        self.node_index = torch.arange(num_nodes, out=LongTensor())

        # uniform sampling on the sphere using a repulsive model
        x, y, z = repulsive_sampling(
            self.nsamples,
            loss_fn=lambda x_: repulsive_loss(x_, 1.0, 10.0),
            radius=math.pi,
            device=device,
            max_iter=25000,
        )

        # convert cartesian positions of the nodes on the sphere to beta and gamma rotations
        _, beta, gamma = xyz2alphabetagamma(x, y, z)

        alpha = torch.arange(-math.pi / 2, math.pi / 2, math.pi / self.nalpha)

        self.alpha = alpha.unsqueeze(1).expand(self.nalpha, self.nsamples).flatten()
        self.beta = beta.unsqueeze(0).expand(self.nalpha, self.nsamples).flatten()
        self.gamma = gamma.unsqueeze(0).expand(self.nalpha, self.nsamples).flatten()

    def _initedges(
        self,
        sigmas: Tuple[float, float, float],
        knn: int,
        weight_kernel: Callable,
        kappa: float,
        device: Device,
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

        if not 0.0 <= kappa < 1.0:
            raise ValueError(f"{kappa} is not a valid value for kappa, must be in [0,1).")

        Gg = self.node_Gg(device).reshape(self.num_nodes, -1)
        Gh = self.node_Gg(device).inverse().reshape(self.num_nodes, -1)

        xi = Vi(Gh)  # sources
        xj = Vj(Gg)  # targets

        # Transposed operation for Rodriguez formula
        xi_t = Vi(Gg)
        xj_t = Vj(Gh)

        sqdist = so3_anisotropic_square_riemannanian_distance(xi, xj, xi_t, xj_t, sigmas, device)

        edge_sqdist, neighbors = sqdist.Kmin_argKmin(knn + 1, dim=1)

        edge_index = torch.stack(
            (self.node_index.repeat_interleave(knn + 1), neighbors.cpu().flatten()), dim=0
        )
        edge_sqdist = edge_sqdist.cpu().flatten()

        # remove duplicated edges due to too high knn
        edge_index, edge_sqdist = remove_duplicated_edges(edge_index, edge_sqdist, knn + 1)

        # remove self loops
        edge_index, edge_sqdist = remove_self_loops(edge_index, edge_sqdist)

        # as an heuristic, we choose sigma as the mean squared Riemannian distance
        edge_weight = weight_kernel(edge_sqdist, edge_sqdist.mean())

        # remove directed edges
        edge_index, edge_weight = remove_directed_edges(edge_index, edge_weight, self.num_nodes)

        # compress graph
        if kappa > 0.0:
            edge_index, edge_weight = multinomial_compression(edge_index, edge_weight, kappa)
            edge_index, edge_weight = remove_directed_edges(edge_index, edge_weight, self.num_nodes)

        self.edge_index, self.edge_weight = edge_index, edge_weight

    def _initprojection(self):
        """
        Inits indices to project 2d images on the sphere. The projection is inspired by the Mercator projection.
        """
        self.ix = ((self.gamma + 1) / 2).clamp(0.0, 1.0)
        self.iy = ((-4 * torch.tan(self.beta / 4) + 1) / 2).clamp(0.0, 1.0)
        self.iz = (self.alpha + math.pi / 2) / math.pi

    @property
    def nsym(self) -> int:
        """
        Returns the number of symmetry's layers.

        Returns:
            int: number of symmetry's layers.
        """
        return self.nalpha

    def node_Gg(self, device) -> FloatTensor:
        """
        Returns the matrix formulation of group elements.

        Args:
            device (Device): computation device.

        Returns:
            (FloatTensor): nodes' in matrix formulation
        """
        return so3_matrix(self.alpha, self.beta, self.gamma, device=device)

    @property
    def node_pos(self):
        """
        Returns the cartesian positions of the nodes of the graph.

        Returns:
            (FloatTensor): x nodes' positions.
            (FloatTensor): y nodes' positions.
            (FloatTensor): z nodes' positions.
        """
        return alphabetagamma2xyz(self.alpha, self.beta, self.gamma)

    @property
    def centroid_index(self) -> int:
        """
        Returns the index of the centroid node of the graph.

        Returns:
            (int): centroid node's index.
        """
        return 0

    def project(self, signal):
        """
        Projects a signal on the group equivariant graph.

        Args:
            signal (FloatTensor): input tensor with shape (..., L, H, W).

        Returns:
            (FloatTensor): output tensor with shape (..., L, H, W).
        """
        *_, L, H, W = signal.shape
        return signal[
            ...,
            (self.iz * (L - 1)).long(),
            (self.iy * (H - 1)).long(),
            (self.ix * (W - 1)).long(),
        ]


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
        kappa: Optional[float] = 0.0,
        device=None,
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
        self._initedges(sigmas, knn, weight_kernel, kappa, device)

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

        self.node_index = torch.arange(num_nodes, out=LongTensor())

        x_axis = torch.arange(0.0, self.nx, out=FloatTensor())
        y_axis = torch.arange(0.0, self.ny, out=FloatTensor())
        theta_axis = torch.arange(
            -math.pi / 2, math.pi / 2, math.pi / self.ntheta, out=FloatTensor()
        )

        theta, y, x = torch.meshgrid(theta_axis, y_axis, x_axis)

        self.node_x = x.flatten()
        self.node_y = y.flatten()
        self.node_theta = theta.flatten()

    def _initedges(
        self,
        sigmas: Tuple[float, float, float],
        knn: int,
        weight_kernel: Callable,
        kappa: float,
        device: Device,
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

        if not 0.0 <= kappa < 1.0:
            raise ValueError(f"{kappa} is not a valid value for kappa, must be in [0,1).")

        xi = Vi(torch.inverse(self.node_Gg(device)).reshape(self.num_nodes, -1))  # sources
        xj = Vj(self.node_Gg(device).reshape(self.num_nodes, -1))  # targets

        sqdist = se2_anisotropic_square_riemannanian_distance(xi, xj, sigmas, device)
        edge_sqdist, neighbors = sqdist.Kmin_argKmin(knn + 1, dim=1)

        edge_index = torch.stack(
            (self.node_index.repeat_interleave(knn + 1), neighbors.cpu().flatten()), dim=0
        )
        edge_sqdist = edge_sqdist.cpu().flatten()

        # remove duplicated edges due to too high knn
        edge_index, edge_sqdist = remove_duplicated_edges(edge_index, edge_sqdist, knn + 1)

        # remove self loops
        edge_index, edge_sqdist = remove_self_loops(edge_index, edge_sqdist)

        # as an heuristic, we choose sigma as the mean squared Riemannian distance
        edge_weight = weight_kernel(edge_sqdist, edge_sqdist.mean())

        # remove directed edges
        edge_index, edge_weight = remove_directed_edges(edge_index, edge_weight, self.num_nodes)

        # compress graph
        if kappa > 0.0:
            edge_index, edge_weight = multinomial_compression(edge_index, edge_weight, kappa)
            edge_index, edge_weight = remove_directed_edges(edge_index, edge_weight, self.num_nodes)

        self.edge_index, self.edge_weight = edge_index, edge_weight

    @property
    def nsym(self) -> int:
        """
        Returns the number of symmetry's layers.

        Returns:
            int: number of symmetry's layers.
        """
        return self.ntheta

    def node_Gg(self, device) -> FloatTensor:
        """
        Returns the matrix formulation of group elements.

        Args:
            device (Device): computation device.

        Returns:
            (FloatTensor): nodes' in matrix formulation
        """
        return se2_matrix(self.node_x, self.node_y, self.node_theta, device=device)

    @property
    def node_pos(self) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """
        Return the cartesian positions of the nodes of the graph.

        Returns:
            (FloatTensor): x nodes' positions.
            (FloatTensor): y nodes' positions.
            (FloatTensor): z nodes' positions.
        """
        return self.node_x, self.node_y, self.node_theta

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

    def project(self, signal: FloatTensor) -> FloatTensor:
        """
        Projects a signal on the group equivariant graph.

        Args:
            signal (FloatTensor): input tensor with shape (..., L, H, W).

        Returns:
            (FloatTensor): output tensor with shape (..., L, H, W).
        """
        return signal
