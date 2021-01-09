import math
from typing import Optional, Tuple

import torch
from pykeops.torch import LazyTensor, Pm, Vi, Vj
from scipy.sparse.linalg import ArpackError, eigsh

from ..utils import sparse_tensor_to_sparse_array
from .compression import edge_compression, node_compression
from .signal_processing import get_fourier_basis, get_laplacian
from .utils import delta_pos, metric_tensor, process_edges, remove_self_loops, square_distance, to_undirected


class HyperCubeGraph:
    def __init__(
        self,
        grid_size: Tuple[float, float],
        nx3: Optional[int] = 6,
        kappa: Optional[float] = 0.0,
        weight_kernel: Optional[Tuple[str, float]] = None,
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
            weight_kernel (dict, optional): weight kernel to use, in format {'name': value, 'sigma': value}. Defaults to None.
            knn (int, optional): maximum number of connections of a vertex. Defaults to 16.
            sigmas (Optional, optional): anisotropy parameters. Defaults to (1.0, 1.0, 1.0).
        """

        weight_kernel = weight_kernel or ("gaussian", 1.0)

        self.nx1, self.nx2 = grid_size
        self.nx3 = nx3

        self._initnodes(self.nx1 * self.nx2 * self.nx3)
        print("Nodes: Done!")

        self._initedges(sigmas, knn, weight_kernel, kappa)
        print("Edges: Done!")

        self._initlaplacian()
        print("Laplacian: Done!")

    def _initnodes(self, num_nodes: int):
        """
        Init node indices and positions (hypercube pose). The stored attributes are:
            - num_nodes (int): number of nodes.
            - xi_axis (torch.tensor): discretization of the i-th axis.
            - node_index (torch.tensor): indices of nodes in format (num_nodes).
            - node_pos (torch.tensor): positions of nodes in format (num_nodes, 3).

        Args:
            num_nodes (int): number of nodes to create.
        """

        self.node_index = torch.arange(num_nodes)

        # we define the grid points and reshape them to get 1-d arrays
        self.x1_axis = torch.arange(0.0, self.nx1)
        self.x2_axis = torch.arange(0.0, self.nx2)
        self.x3_axis = torch.arange(0.0, math.pi, math.pi / self.nx3)

        # we keep in memory the position of all the nodes, before compression
        # easier to deal with indices from here
        x3_, x2_, x1_ = torch.meshgrid(self.x3_axis, self.x2_axis, self.x1_axis)
        self.node_pos = torch.stack([x1_.flatten(), x2_.flatten(), x3_.flatten()], axis=-1)

        self.num_nodes = num_nodes

    def _initedges(self, sigmas: Tuple[float, float, float], knn: int, weight_kernel: Tuple[str, float], kappa: float):
        """
        Init edge indices and attributes (weights). The stored attributes are:
            - edge_index (torch.tensor): indices of edges in format (2, num_edges).
            - edge_weight (torch.tensor): weight of edges in format (num_edges).

        Args:
            sigmas (Tuple[float,float,float]): anisotropic parameters to compute Riemannian distance.
            knn (int): maximum number of connections of a vertex.
            weight_kernel (tuple):  weight kernel to use, in format (name, sigma).
            kappa (float): edges compression rate.

        Raises:
            ValueError: knn must be strictly lower than number of nodes - 1.
            ValueError: weight kernel must be gaussian, laplacian or cauchy.
            ValueError: kappa must be in [0, 1).
        """
        if knn > self.num_nodes - 1:
            raise ValueError(f"{knn} is not a valid value for KNN graph with {self.num_nodes} nodes")

        w_kernel, w_sigma = weight_kernel
        if w_kernel not in ["gaussian", "laplacian", "cauchy"]:
            raise ValueError(
                f"{w_kernel} is not a valid value for w_kernel, it must be 'gaussian', 'laplacian' or 'cauchy'"
            )

        if not 0.0 <= kappa < 1.0:
            raise ValueError(f"{kappa} is not a valid value for kappa, must be in [0,1).")

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        xi = Vi(self.node_pos.to(device))  # sources
        xj = Vj(self.node_pos.to(device))  # targets

        S = metric_tensor(sigmas, device)

        edge_sqdist, neighbors = square_distance(xi, xj, S).Kmin_argKmin(knn + 1, dim=0)

        edge_index = torch.stack((self.node_index.repeat_interleave(knn + 1), neighbors.cpu().flatten()), dim=0)
        edge_sqdist = edge_sqdist.cpu().flatten()

        edge_index, edge_sqdist = process_edges(edge_index, edge_sqdist, kappa)

        if w_kernel == "gaussian":
            kernel = lambda sqdistc, sigmac: torch.exp(-sqdistc / sigmac ** 2)
        elif w_kernel == "laplacian":
            kernel = lambda sqdistc, sigmac: torch.exp(-torch.sqrt(sqdistc) / sigmac)
        elif w_kernel == "cauchy":
            kernel = lambda sqdistc, sigmac: 1 / (1 + sqdistc / sigmac ** 2)

        edge_weight = kernel(edge_sqdist, w_sigma)

        self.edge_index, self.edge_weight = edge_index, edge_weight

    def _initlaplacian(self):
        """
        Init the symmetric normalized laplacian of the graph. The stored attributes are:
            - laplacian (torch.sparse.tensor): symmetric normalized laplacian.
            - lmax (float): maximum eigenvalue of the laplacian.
        """
        self.laplacian = get_laplacian(self.edge_index, self.edge_weight, norm="sym", num_nodes=self.num_nodes)

        try:
            lmax = eigsh(sparse_tensor_to_sparse_array(self.laplacian), k=1, which="LM", return_eigenvectors=False)
            self.lmax = float(lmax.real)
        except ArpackError:
            # in case the eigen decomposition's algorithm does not converge, set lmax to theoretic upper bound.
            self.lmax = 2.0

    def neighborhood(self, node_idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Return neighborhood of a given node.

        Args:
            node_idx (int): node index.
            return_weights (bool, optional): indicator if edges weight are returned. Defaults to True.

        Returns:
            (torch.tensor): neighbours index of node.
            (torch.tensor): neighbours weight of edges.
        """
        mask = self.edge_index[0] == node_idx
        neighbors = self.edge_index[1, mask]

        weights = self.edge_weight[mask]
        return neighbors, weights

    @property
    def fourier_basis(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        return get_fourier_basis(self.laplacian)

    @property
    def num_edges(self) -> int:
        """
        Return the total number of edges of the graph.

        Returns:
            (int): number of edges.
        """
        return self.edge_index.shape[1]

    @property
    def centroid_index(self) -> int:
        """
        Return the index of the centroid node of the graph.

        Returns:
            (int): centroid node index.
        """
        return int(self.nx1 / 2) + int(self.nx2 / 2) * self.nx1 + int(self.nx3 / 2) * self.nx1 * self.nx2
