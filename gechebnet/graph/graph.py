import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from pykeops.torch import LazyTensor, Pm, Vi, Vj
from scipy.sparse.linalg import ArpackError, eigsh
from torch import FloatTensor, LongTensor
from torch.sparse import FloatTensor as SparseFloatTensor

from ..utils import sparse_tensor_to_sparse_array
from .riemannian import metric_tensor, sq_riemannian_distance
from .signal_processing import get_fourier_basis, get_laplacian
from .utils import process_edges


class Graph:
    def __init__(self, *arg, **kwargs):
        """
        Init the graph attributes with empty tensors
        """
        self.node_index = LongTensor()
        self.edge_index = LongTensor()
        self.edge_weight = FloatTensor()
        self.laplacian = SparseFloatTensor()

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

    @property
    def fourier_basis(self) -> Tuple[ndarray, ndarray]:
        """
        Return graph Fourier basis, i.e. Laplacian eigen decomposition.

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
        lambdas, Phi = self.fourier_basis
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


class SE2GEGraph(Graph):
    def __init__(
        self,
        grid_size: Tuple[int, int],
        nx3: Optional[int] = 6,
        kappa: Optional[float] = 0.0,
        weight_kernel: Optional[Callable] = None,
        sigma_fn: Optional[Callable] = None,
        knn: Optional[int] = 16,
        sigmas: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    ):
        """
        Init a SE(2) group equivariant graph with `nx3` equivariance layers:
            1. Init nodes on the hyper cube.
            2. Init edges between nodes. Each node has at most knn neighbors, weight of edges are computed according to the
            Riemannian distance between them and the given weight kernel.
            3. Compress the graph according to the given compression algorithm.
            4. Init laplacian the symmetric normalized laplacian of the graph and store its maximum eigen value.

        Args:
            grid_size (tuple): spatial dimension in format (nx1, nx2).
            nx3 (int, optional): number of equivariance's layers. Defaults to 6.
            kappa (float, optional): edges compression rate. Defaults to 0.0.
            weight_kernel (callable, optional): weight kernel to use. Defaults to None.
            knn (int, optional): maximum number of connections of a vertex. Defaults to 16.
            sigmas (Optional, optional): anisotropy parameters. Defaults to (1.0, 1.0, 1.0).
        """

        super().__init__()

        if weight_kernel is None:
            weight_kernel = lambda sqdistc, sigmac: torch.exp(-sqdistc / sigmac ** 2)

        self.nx1, self.nx2 = grid_size
        self.nx3 = nx3

        self._initnodes(self.nx1 * self.nx2 * self.nx3)
        self._initedges(sigmas, knn, weight_kernel, kappa)
        self._initlaplacian()

    def _initnodes(self, num_nodes: int):
        """
        Init node indices and positions (hypercube pose). The stored attributes are:
            - xi_axis (FloatTensor): discretization of the i-th axis.
            - node_index (LongTensor): indices of nodes in format (num_nodes).
            - node_pos (FloatTensor): positions of nodes in format (num_nodes, 3).

        Args:
            num_nodes (int): number of nodes to create.
        """

        self.node_index = torch.arange(num_nodes, out=LongTensor())

        # we define the grid points and reshape them to get 1-d arrays
        self.x1_axis = torch.arange(0.0, self.nx1, out=FloatTensor())
        self.x2_axis = torch.arange(0.0, self.nx2, out=FloatTensor())
        self.x3_axis = torch.arange(0.0, math.pi, math.pi / self.nx3, out=FloatTensor())

        # we keep in memory the position of all the nodes, before compression
        # easier to deal with indices from here
        x3_, x2_, x1_ = torch.meshgrid(self.x3_axis, self.x2_axis, self.x1_axis)
        self.node_pos = torch.stack([x1_.flatten(), x2_.flatten(), x3_.flatten()], axis=-1)

    def _initedges(
        self,
        sigmas: Tuple[float, float, float],
        knn: int,
        weight_kernel: Callable,
        kappa: float,
    ):
        """
        Init edge indices and attributes (weights). The stored attributes are:
            - edge_index (LongTensor): indices of edges in format (2, num_edges).
            - edge_weight (FloatTensor): weight of edges in format (num_edges).

        Args:
            sigmas (Tuple[float,float,float]): anisotropic parameters to compute Riemannian distance.
            knn (int): maximum number of connections of a vertex.
            weight_kernel (tuple):  weight kernel to use, in format (name, sigma).
            kappa (float): edges compression rate.

        Raises:
            ValueError: kappa must be in [0, 1).
        """

        if not 0.0 <= kappa < 1.0:
            raise ValueError(f"{kappa} is not a valid value for kappa, must be in [0,1).")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        xi = Vi(self.node_pos.to(device))  # sources
        xj = Vj(self.node_pos.to(device))  # targets

        S = metric_tensor(sigmas, device)

        edge_sqdist, neighbors = sq_riemannian_distance(xi, xj, S).Kmin_argKmin(knn + 1, dim=0)

        edge_index = torch.stack(
            (self.node_index.repeat_interleave(knn + 1), neighbors.cpu().flatten()), dim=0
        )
        edge_sqdist = edge_sqdist.cpu().flatten()

        # as an heuristic, we choose sigma as the mean squared Riemannian distance
        edge_weight = weight_kernel(edge_sqdist, edge_sqdist.mean())
        edge_index, edge_weight = process_edges(edge_index, edge_weight, knn + 1, kappa)

        self.edge_index, self.edge_weight = edge_index, edge_weight

    def _initlaplacian(self):
        """
        Init the symmetric normalized laplacian of the graph. The stored attributes are:
            - laplacian (torch.sparse.tensor): symmetric normalized laplacian.
            - lmax (float): maximum eigenvalue of the laplacian.
        """
        self.laplacian = get_laplacian(self.edge_index, self.edge_weight, self.num_nodes)

        try:
            lmax = eigsh(
                sparse_tensor_to_sparse_array(self.laplacian),
                k=1,
                which="LM",
                return_eigenvectors=False,
            )
            self.lmax = float(lmax.real)
        except ArpackError:
            # in case the eigen decomposition's algorithm does not converge, set lmax to theoretic upper bound.
            self.lmax = 2.0

    @property
    def centroid_index(self) -> int:
        """
        Return the index of the centroid node of the graph.

        Returns:
            (int): centroid node index.
        """
        return (
            int(self.nx1 / 2)
            + int(self.nx2 / 2) * self.nx1
            + int(self.nx3 / 2) * self.nx1 * self.nx2
        )
