import math

import torch
from pykeops.torch import LazyTensor, Pm, Vi, Vj
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import remove_self_loops

from ..utils import sparse_tensor_to_sparse_array
from .compression import edge_compression, node_compression
from .signal_processing import get_laplacian
from .utils import metric_tensor


class Graph:
    def _initlaplacian(self):
        self.laplacian = get_laplacian(self.edge_index, self.edge_weight, norm="sym", num_nodes=self.num_nodes)
        lmax = eigsh(sparse_tensor_to_sparse_array(self.laplacian), k=1, which="LM", return_eigenvectors=False)
        self.lmax = float(lmax.real)

    def _graphcompression(self, compression):
        if compression is not None:
            if compression["type"] == "node":
                self = node_compression(self, compression["kappa"])
            elif compression["type"] == "edge":
                self = edge_compression(self, compression["kappa"])

    @property
    def num_edges(self):
        return self.edge_index.shape[1]

    def neighborhood(self, node_idx, return_weights=True):
        mask = self.edge_index[0] == node_idx
        neighbors = self.edge_index[1, mask]

        if not return_weights:
            return neighbors

        weights = self.edge_weight[mask]
        return neighbors, weights


class HyperCubeGraph(Graph):
    def __init__(
        self,
        grid_size,
        nx3=6,
        compression=None,
        self_loop=False,
        weight_kernel="gaussian",
        weight_sigma=1.0,
        knn=26,
        sigmas=(1.0, 1.0, 1.0),
        weight_comp_device=None,
    ):
        """
        Initialise the GraphData object:
            - init the nodes and positions
            - init the edges and weights

        Args:
            grid_size (tuple): the spatial dimension of the graph in format (nx1, nx2).
            num_layers (int, optional): the orientation dimension of the graph. Defaults to 6.
            static_compression (dict, optional): the static compression algorithm to reduce
                the graph size. Either ("node", kappa) or ("edge", kappa). Defaults to None.
            self_loop (bool, optional): the indicator if the graph can contains self-loop. Defaults to True.
            weight_kernel (WeightKernel, optional): the weights' kernel. Defaults to None.
            dist_threshold (float, optional): the maximum distance between two nodes to be linked. Defaults to 1.0.
            sigmas (tuple, optional): the anisotropic intensities. Defaults to (1.0, 1.0, 1.0).
            batch_size (int, optional): the batch size when computing edges' weights. Defaults to 1000.
        """

        weight_comp_device = weight_comp_device or torch.device("cpu")

        if not self_loop:
            knn += 1

        self.nx1, self.nx2 = grid_size
        self.nx3 = nx3
        self._initnodes()
        print("Nodes: Done!")

        self._initedges(sigmas, knn, weight_kernel, weight_sigma, self_loop, weight_comp_device)
        print("Edges: Done!")

        super()._graphcompression(compression)
        print("Compression: Done!")

        super()._initlaplacian()
        print("Laplacian: Done!")

    def _initnodes(self):
        """
        Initialise the nodes' indices and their positions.
            - `self.node_index` is a tensor with shape (num_nodes)
            - `self.node_pos` is a tensor with shape (self.nx1 * self.nx2 * self.nx3, 3)
        If the compression algorithm is the static node compression, remove a proportion kappa of nodes.
        """
        self.node_index = torch.arange(self.nx1 * self.nx2 * self.nx3)

        # we define the grid points and reshape them to get 1-d arrays
        self.x1_axis = torch.arange(0.0, self.nx1, 1.0)
        self.x2_axis = torch.arange(0.0, self.nx2, 1.0)
        self.x3_axis = torch.arange(0.0, math.pi, math.pi / self.nx3)

        # we keep in memory the position of all the nodes, before compression
        # easier to deal with indices from here
        x3_, x2_, x1_ = torch.meshgrid(self.x3_axis, self.x2_axis, self.x1_axis)
        self.node_pos = torch.stack([x1_.flatten(), x2_.flatten(), x3_.flatten()], axis=-1)

    def _initedges(self, sigmas, knn, weight_kernel, weight_sigma, self_loop, device):
        node_pos = self.node_pos.to(device)

        xi = Vi(node_pos)
        xj = Vj(node_pos)
        Sj = Vj(metric_tensor(node_pos, sigmas, device))
        pi = Pm([torch.tensor([math.pi], device=device)])

        dx1 = (xi[0] - xj[0]).abs()
        dx2 = (xi[1] - xj[1]).abs()
        dx3 = LazyTensor.cat(((xi[2] - xj[2]).abs(), pi - (xi[2] - xj[2]).abs()), dim=-1).min()  # periodicity x3_axis

        dx = LazyTensor.cat((dx1, dx2, dx3), dim=-1)

        if weight_kernel == "gaussian":
            kernel = lambda dxc, Sc, sigmac: (-dxc.weightedsqnorm(Sc) / (sigmac ** 2)).exp()
        elif weight_kernel == "laplacian":
            kernel = lambda dxc, Sc, sigmac: (-(dxc.weightedsqnorm(Sc).sqrt() / (sigmac))).exp()
        elif weight_kernel == "cauchy":
            kernel = lambda dxc, Sc, sigmac: (1 + dxc.weightedsqnorm(Sc) / (sigmac ** 2)).power(-1)

        weights_tensor = kernel(dx, Sj, weight_sigma)

        neg_weights, neighbors = (-weights_tensor).Kmin_argKmin(knn, dim=0)

        edge_index = torch.stack((self.node_index.repeat_interleave(knn), neighbors.flatten().cpu()), dim=0)
        edge_weight = -neg_weights.cpu().flatten()

        self.process_edges(edge_index, edge_weight, self_loop)

    def process_edges(self, edge_index, edge_weight, self_loop):
        if not self_loop:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # remove edges at the boundaries of the images, based on the minimal weight of the centroid node's neighborhood
        mask_neighborhood = edge_index[0] == self.centroid_index
        mask_threshold = edge_weight >= edge_weight[mask_neighborhood].min()
        self.edge_index, self.edge_weight = edge_index[:, mask_threshold], edge_weight[mask_threshold]

    @property
    def centroid_index(self):
        return int(self.nx1 / 2) + int(self.nx2 / 2) * self.nx1 + int(self.nx3 / 2) * self.nx1 * self.nx2

    @property
    def num_nodes(self):
        return self.nx1 * self.nx2 * self.nx3
