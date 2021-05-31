# coding=utf-8

import hashlib
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from ..geometry.se import r2_matrix, r2_riemannian_sqdist, r2_uniform_sampling, se2_matrix, se2_riemannian_sqdist, se2_uniform_sampling
from ..geometry.so import s2_matrix, s2_riemannian_sqdist, s2_uniform_sampling, so3_matrix, so3_riemannian_sqdist, so3_uniform_sampling
from ..geometry.utils import betagamma2xyz, xyz2betagamma
from ..utils.utils import round
from .gsp import get_fourier_basis, get_laplacian, get_rescaled_laplacian
from .utils import to_undirected


class Graph:
    def get_laplacian(self, rescale=True, device=None):
        """
        Returns symmetric normalized graph laplacian .

        Args:
            rescale (bool, optional): if True, it returns rescaled normalized laplacian with eigenvalues
                in [-1, 1]. Defaults to True.
            device (`torch.device`, optional): computation device. Defaults to None.

        Returns:
            (`torch.sparse.FloatTensor`): laplacian.
        """
        if not hasattr(self, "laplacian"):
            if rescale:
                self.laplacian = get_rescaled_laplacian(
                    self.edge_index, self.edge_weight, self.num_vertices, 2.0, device
                )
            else:
                self.laplacian = get_laplacian(self.edge_index, self.edge_weight, self.num_vertices, device=device)

        return self.laplacian

    def get_eigen_space(self, norm=False):
        """
        Return graph eigen space, i.e. Laplacian eigen decomposition.

        Args:
            norm (bool, optional): if True, uses the normalized laplacian with eigenvalues in [-1, 1]. Defaults to False.

        Returns:
            (`np.ndarray`): Laplacian eigen values.
            (`np.ndarray`): Laplacian eigen vectors.
        """
        if not hasattr(self, "eigen_space"):
            self.eigen_space = get_fourier_basis(self.get_laplacian(norm))

        return self.eigen_space

    def diff_kernel(self, kernel):
        """
        Return the diffusion kernel of the graph specified by the kernel input.

        Args:
            kernel (callable): diffusion kernel.

        Returns:
            (`np.ndarray`): diffusion kernel.
        """
        lambdas, Phi = self.get_eigen_space()
        return Phi @ np.diag(kernel(lambdas)) @ Phi.T

    @property
    def num_vertices(self):
        """
        Return the total number of vertices of the graph.

        Returns:
            (int): number of vertices.
        """
        return self.vertex_index.shape[0]

    @property
    def num_edges(self):
        """
        Return the total number of edges of the graph.

        Returns:
            (int): number of (directed) edges.
        """
        return self.edge_index.shape[1]

    def neighborhood(self, vertex_index):
        """
        Returns the vertex's neighborhood.

        Args:
            vertex_index (int): vertex index.

        Returns:
            (`torch.LongTensor`): neighbors' indices.
            (`torch.FloatTensor`): weights of edges from vertex to neighbors.
            (`torch.FloatTensor`): squared riemannian distance from vertex to neighbors.
        """
        if not vertex_index in self.vertex_index:
            raise ValueError(f"{vertex_index} is not a valid vertex index")

        mask = self.edge_index[0] == vertex_index
        return self.edge_index[1, mask], self.edge_weight[mask], self.edge_sqdist[mask]

    @property
    def is_connected(self):
        """
        Returns:
            (bool): True if the graph is connected, i.e. it does not contain isolated vertex.
        """
        return torch.allclose(self.edge_index.unique(), self.vertex_index.unique())

    @property
    def is_undirected(self):
        """
        Returns:
            (bool): True if the graph is undirected.
        """
        matrix = torch.sparse.FloatTensor(
            self.edge_index, self.edge_weight, torch.Size((self.num_vertices, self.num_vertices))
        )
        return ((matrix - matrix.t()).coalesce().values().abs().sum() < 1e-6).item()

    def save(self, path_to_graph):
        """
        Save graph's attributes.
        """

        os.makedirs(path_to_graph, exist_ok=True)

        torch.save(self.vertex_index, os.path.join(path_to_graph, f"{self.hash_repr()}_vertex_index.pt"))

        torch.save(self.edge_index, os.path.join(path_to_graph, f"{self.hash_repr()}_edge_index.pt"))
        torch.save(self.edge_sqdist, os.path.join(path_to_graph, f"{self.hash_repr()}_edge_sqdist.pt"))
        torch.save(self.edge_weight, os.path.join(path_to_graph, f"{self.hash_repr()}_edge_weight.pt"))

        for vertex_attr in self.vertex_attributes:
            torch.save(getattr(self, vertex_attr), os.path.join(path_to_graph, f"{self.hash_repr()}_{vertex_attr}.pt"))

    def load(self, path_to_graph):
        """
        Load graph's attributes.
        """

        self.vertex_index = torch.load(os.path.join(path_to_graph, f"{self.hash_repr()}_vertex_index.pt"))

        self.edge_index = torch.load(os.path.join(path_to_graph, f"{self.hash_repr()}_edge_index.pt"))
        self.edge_sqdist = torch.load(os.path.join(path_to_graph, f"{self.hash_repr()}_edge_sqdist.pt"))
        self.edge_weight = torch.load(os.path.join(path_to_graph, f"{self.hash_repr()}_edge_weight.pt"))

        for vertex_attr in self.vertex_attributes:
            setattr(self, vertex_attr, torch.load(os.path.join(path_to_graph, f"{self.hash_repr()}_{vertex_attr}.pt")))

    def hash_repr(self):
        return hashlib.sha256(self.str_repr.encode("utf-8")).hexdigest()

    def check_graph_exists(self, path_to_graph):
        """
        Returns:
            (bool): True if the graph exists in the graphs' directory.
        """

        if not os.path.exists(os.path.join(path_to_graph, f"{self.hash_repr()}_vertex_index.pt")):
            return False

        if not os.path.exists(os.path.join(path_to_graph, f"{self.hash_repr()}_edge_index.pt")):
            return False

        if not os.path.exists(os.path.join(path_to_graph, f"{self.hash_repr()}_edge_weight.pt")):
            return False

        if not os.path.exists(os.path.join(path_to_graph, f"{self.hash_repr()}_edge_sqdist.pt")):
            return False

        for vertex_attr in self.vertex_attributes:
            if not os.path.exists(os.path.join(path_to_graph, f"{self.hash_repr()}_{vertex_attr}.pt")):
                return False

        return True


class RandomSubGraph(Graph):
    def __init__(self, graph):
        """
        Args:
            graph (`Graph`): parent graph.
        """

        self.graph = graph
        self.manifold = self.graph.manifold

        self.vertex_index = self.graph.vertex_index.clone()
        self.sub_vertex_index = self.vertex_index.clone()

        for attr in self.graph.vertex_attributes:
            setattr(self, attr, getattr(graph, attr))

        self.edge_index = self.graph.edge_index.clone()
        self.edge_weight = self.graph.edge_weight.clone()
        self.edge_sqdist = self.graph.edge_sqdist.clone()

    def reinit(self):
        """
        Reinitialize random sub-graph vertices and edges' attributes.
        """
        print("Reinit graph...")
        if hasattr(self, "laplacian"):
            del self.laplacian

        self.vertex_index = self.graph.vertex_index.clone()
        self.sub_vertex_index = self.vertex_index.clone()

        for attr in self.graph.vertex_attributes:
            setattr(self, attr, getattr(self.graph, attr))

        self.edge_index = self.graph.edge_index.clone()
        self.edge_weight = self.graph.edge_weight.clone()
        self.edge_sqdist = self.graph.edge_sqdist.clone()
        print("Done!")

    def edges_sampling(self, rate):
        """
        Randomly samples a given rate of edges from the original graph to generate a random sub-graph.
        The graph is assumed to be undirected and the probability for an edge to be sampled is proportional
        to its weight.

        Args:
            rate (float): rate of edges to sample.
        """
        print("Sample edges...")
        mask = self.graph.edge_index[0] < self.graph.edge_index[1]
        edge_index = self.graph.edge_index[..., mask]
        edge_weight = self.graph.edge_weight[mask]
        edge_sqdist = self.graph.edge_sqdist[mask]

        num_samples = math.ceil(rate * edge_weight.nelement())
        sampled_edges = torch.multinomial(edge_weight, num_samples)

        sampled_edge_index = edge_index[..., sampled_edges]
        sampled_edge_weight = edge_weight[sampled_edges]
        sampled_edge_sqdist = edge_sqdist[sampled_edges]

        self.edge_index = torch.cat((sampled_edge_index.flip(0), sampled_edge_index), 1)
        self.edge_weight = sampled_edge_weight.repeat(2)
        self.edge_sqdist = sampled_edge_sqdist.repeat(2)
        print("Done!")

    def vertices_sampling(self, rate):
        """
        Randomly samples a given rate of vertices from the original graph to generate a random subgraph.
        All the vertices have the same probability being sampled and at the end of the algorithm, it only remains
        edges between sampled vertices.

        Warning: vertices' sampling is not compatible with pooling and unpooling operations for now!!!

        Args:
            rate (float): rate of vertices to sample.
        """
        print("Sample vertices...")
        # samples N vertices from the original graph
        num_samples = math.floor(rate * self.graph.num_vertices)
        sampled_vertices, _ = torch.multinomial(torch.ones(self.graph.num_vertices), num_samples).sort()

        self.vertex_index = torch.arange(num_samples)

        self.sub_vertex_index = sampled_vertices.clone()

        for attr in self.graph.vertex_attributes:
            setattr(self, attr, getattr(self.graph, attr)[sampled_vertices])

        # selects edges between sampled vertices and resets the edge indices with the current vertex mapping
        vertex_mapping = torch.empty(self.graph.num_vertices, dtype=torch.long).fill_(-1)
        vertex_mapping[self.graph.vertex_index[sampled_vertices]] = self.vertex_index
        edge_index = vertex_mapping[self.graph.edge_index]
        mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
        self.edge_index = edge_index[:, mask]
        self.edge_weight = self.graph.edge_weight[mask]
        self.edge_sqdist = self.graph.edge_sqdist[mask]
        print("Done!")

    def cartesian_pos(self, axis=None):
        """
        Returns the cartesian position of the graph's vertices.

        Args:
            axis (str, optional): cartesian axis. If None, return all axis. Defaults to None.

        Returns:
            (`torch.FloatTensor`, optional): x position.
            (`torch.FloatTensor`, optional): y position.
            (`torch.FloatTensor`, optional): z position.
        """
        x, y, z = self.graph.cartesian_pos()

        if axis == "x":
            return x[self.sub_vertex_index]
        if axis == "y":
            return y[self.sub_vertex_index]
        if axis == "z":
            return z[self.sub_vertex_index]
        return x[self.sub_vertex_index], y[self.sub_vertex_index], z[self.sub_vertex_index]

    @property
    def vertex_attributes(self):
        """
        Returns the graph's vertices attributes.

        Returns:
            (tuple): tuple of vertices' attributes
        """
        return self.graph.vertex_attributes


class GEGraph(Graph):
    """
    Basic class for an (anisotropic) group equivariant graph.
    """

    def __init__(self, size, sigmas, K, path_to_graph, kernel="gaussian"):
        """
        Args:
            uniform_sampling (`torch.Tensor`): uniform sampling on the group manifold in format (D, V) where
                V corresponds to the number of samples points and D to the dimension of the group.
            sigmas (tuple of floats): anisotropy's coefficients.
            K (int): number of neighbors per vertex.
        """
        super().__init__()

        self.size = size
        self.str_repr = f"{self.manifold}-{self.size}-{K}-{sigmas}-{kernel}"
        self._initkernel(kernel)

        if self.check_graph_exists(path_to_graph):
            print("Graph already exists: LOADING...")
            self.load(path_to_graph)
            print("Done!")

        else:
            print("Graph does not already exist: INITIALIZATION...")
            self._initvertices(size)
            self._initedges(sigmas, K)
            print("Done!")
            self.save(path_to_graph)
            print("Saved!")

    def _initkernel(self, kernel):
        """
        [summary]

        Args:
            kernel ([type]): [description]
        """
        if kernel not in {"gaussian", "cauchy", "laplace", "rectangular"}:
            raise ValueError(f"{kernel} is not a valid value for kernel")

        if kernel == "gaussian":
            self.kernel = lambda sq_dist, w: torch.exp(-sq_dist / w)
        elif kernel == "cauchy":
            self.kernel = lambda sq_dist, w: (1 / (1 + -sq_dist / w))
        elif kernel == "laplace":
            self.kernel = lambda sq_dist, w: torch.exp(-torch.sqrt(sq_dist / w))
        elif kernel == "rectangular":
            self.kernel = lambda sq_dist, w: torch.heaviside(sq_dist / w, torch.zeros(1))

    def _initvertices(self, size):
        """
        Args:
            uniform_sampling (tuple of `torch.Tensor`): tuple of uniform sampling on the group manifold, one element
                of the tuple correspond to one dimension.
        """

        samples = self.uniform_sampling(size)
        self.vertex_index = torch.arange(len(samples[0]))

        for dim, dim_samples in zip(self.group_dim, samples):
            setattr(self, f"vertex_{dim}", dim_samples)

    def _initedges(self, sigmas, K):
        """
        Args:
            sigmas (tuple): anisotropy's coefficients.
            K (int): number of neighbors per vertex.
        """
        Gg = self.general_linear_group_element
        edge_sqdist = torch.empty(self.num_vertices * (K + 1))
        edge_index = torch.empty((2, self.num_vertices * (K + 1)), dtype=torch.long)
        Re = torch.diag(torch.tensor(sigmas))

        # compute all pairwise distances of the graph. WARNING: can possibly take a lot of time!!
        for idx in tqdm(self.vertex_index, file=sys.stdout):
            sqdist = round(self.riemannian_sqdist(Gg[idx], Gg, Re), 6)
            values, indices = torch.topk(sqdist, largest=False, k=K + 1, sorted=False)
            edge_sqdist[idx * (K + 1) : (idx + 1) * (K + 1)] = values
            edge_index[0, idx * (K + 1) : (idx + 1) * (K + 1)] = idx
            edge_index[1, idx * (K + 1) : (idx + 1) * (K + 1)] = indices

        # make the graph undirected and avoid asymetries at the boundaries using a maximum squared distance
        # between connected vertices
        mask = edge_index[0] == self.centroid_vertex
        max_sqdist = edge_sqdist[mask].max()
        self.edge_index, self.edge_sqdist = to_undirected(
            edge_index, edge_sqdist, None, self.num_vertices, max_sqdist, self_loop=False
        )

        # the kernel width is proportional to the mean squared distance between connected vertices
        kernel_width = 0.8 * self.edge_sqdist.mean()

        self.edge_weight = self.kernel(self.edge_sqdist, kernel_width)


class SE2GEGraph(GEGraph):
    """
    Approximation of the SE(2) group manifold. The vertices correspond to group elements and the anisotropic
    Riemannian distances are encoded in the graph's edges.
    """

    def __init__(self, size, sigmas, K, path_to_graph, kernel="gaussian"):
        """
        Args:
            size (list of ints): size of the grid in format (nx, ny,ntheta).
            sigmas (tuple of floats): anisotropy's coefficients.
            K (int): number of neighbors.
            path_to_graph (str): path to the folder to save graph.
        """
        self.manifold = "se2"

        if len(size) != 3:
            raise ValueError(f"size must be 3-dimensional")

        if len(sigmas) != 3:
            raise ValueError(f"sigmas must be 3-dimensional")

        super().__init__(size, sigmas, K, path_to_graph, kernel)

    def uniform_sampling(self, size):
        return se2_uniform_sampling(*size)

    @property
    def centroid_vertex(self):
        return self.size[0] // 2 + self.size[1] // 2 * self.size[0] + self.size[2] // 2 * self.size[0] * self.size[1]

    def riemannian_sqdist(self, Gg, Gh, Re):
        """
        Return the riemannian squared distance between GL(3) group elements Gg and Gh according to the
        riemannian metric Re.

        Args:
            Gg (`torch.FloatTensor`): GL(3) group elements.
            Gh (`torch.FloatTensor`): GL(3) group elements.
            Re (`torch.FloatTensor`): riemannian metric.

        Returns:
            (`torch.FloatTensor`): squared riemannian distance.
        """
        return se2_riemannian_sqdist(Gg, Gh, Re)

    @property
    def group_element(self):
        """
        Return the group elements of graph's vertices.

        Returns:
            (`torch.FloatTensor`): SE(2) group's elements.
        """
        return self.vertex_x, self.vertex_y, self.vertex_theta

    @property
    def general_linear_group_element(self):
        """
        Return the general linear group elements of graph's vertices.

        Returns:
            (`torch.FloatTensor`): GL(3) group's elements.
        """
        return se2_matrix(self.vertex_x, self.vertex_y, self.vertex_theta)

    @property
    def group_dim(self):
        """
        Return the name of the group's dimensions.

        Returns:
            (dict): mapping from dimensions' names to dimensions' indices.
        """
        return ["x", "y", "theta"]

    @property
    def dim(self):
        return self.size[2], self.size[0] * self.size[1]

    def cartesian_pos(self, axis=None):
        """
        Return the cartesian positions of the graph's vertices.

        Args:
            axis (str, optional): cartesian axis. If None, return all axis. Defaults to None.

        Returns:
            (`torch.FloatTensor`, optional): x positions.
            (`torch.FloatTensor`, optional): y positions.
            (`torch.FloatTensor`, optional): z positions.
        """
        if axis is None:
            return self.vertex_x, self.vertex_y, self.vertex_theta
        if axis == "x":
            return self.vertex_x
        if axis == "y":
            return self.vertex_y
        if axis == "z":
            return self.vertex_theta

    @property
    def vertex_attributes(self):
        """
        Returns the graph's vertices attributes.

        Returns:
            (tuple): tuple of vertices' attributes
        """
        return ("vertex_x", "vertex_y", "vertex_theta")


class R2GEGraph(GEGraph):
    """
    Approximation of the R(2) manifold. The vertices correspond to manifold elements and the anisotropic
    Riemannian distances are encoded in the graph's edges.
    """

    def __init__(self, size, sigmas, K, path_to_graph, kernel="gaussian"):
        """
        Args:
            size (list of ints): size of the grid in format (nx, ny,ntheta).
            sigmas (tuple of floats): anisotropy's coefficients.
            K (int): number of neighbors.
            path_to_graph (str): path to the folder to save graph.
        """
        self.manifold = "r2"

        if len(size) != 3:
            raise ValueError(f"size must be 3-dimensional")

        if len(sigmas) != 3:
            raise ValueError(f"sigmas must be 3-dimensional")

        super().__init__(size, sigmas, K, path_to_graph, kernel)

    def uniform_sampling(self, size):
        return r2_uniform_sampling(size[0], size[1])

    @property
    def centroid_vertex(self):
        return self.size[0] // 2 + self.size[1] // 2 * self.size[0]

    def riemannian_sqdist(self, Gg, Gh, Re):
        """
        Return the riemannian squared distance between GL(3) group elements Gg and Gh according to the
        riemannian metric Re.

        Args:
            Gg (`torch.FloatTensor`): GL(3) group elements.
            Gh (`torch.FloatTensor`): GL(3) group elements.
            Re (`torch.FloatTensor`): riemannian metric.

        Returns:
            (`torch.FloatTensor`): squared riemannian distance.
        """
        return r2_riemannian_sqdist(Gg, Gh, Re)

    @property
    def group_element(self):
        """
        Return the group elements of graph's vertices.

        Returns:
            (`torch.FloatTensor`): SE(2) group's elements.
        """
        return self.vertex_x, self.vertex_y

    @property
    def general_linear_group_element(self):
        """
        Return the general linear group elements of graph's vertices.

        Returns:
            (`torch.FloatTensor`): GL(3) group's elements.
        """
        return r2_matrix(self.vertex_x, self.vertex_y)

    @property
    def group_dim(self):
        """
        Return the name of the group's dimensions.

        Returns:
            (dict): mapping from dimensions' names to dimensions' indices.
        """
        return ["x", "y"]

    @property
    def dim(self):
        return 1, self.size[0] * self.size[1]

    def cartesian_pos(self, axis=None):
        """
        Return the cartesian positions of the graph's vertices.

        Args:
            axis (str, optional): cartesian axis. If None, return all axis. Defaults to None.

        Returns:
            (`torch.FloatTensor`, optional): x positions.
            (`torch.FloatTensor`, optional): y positions.
            (`torch.FloatTensor`, optional): z positions.
        """
        if axis is None:
            return self.vertex_x, self.vertex_y, torch.zeros(self.num_vertices)
        if axis == "x":
            return self.vertex_x
        if axis == "y":
            return self.vertex_y
        if axis == "z":
            return torch.zeros(self.num_vertices)

    @property
    def vertex_attributes(self):
        """
        Returns the graph's vertices attributes.

        Returns:
            (tuple): tuple of vertices' attributes
        """
        return ("vertex_x", "vertex_y")


class SO3GEGraph(GEGraph):
    """
    Approximation of the SO(3) group manifold. The vertices correspond to group elements and the anisotropic
    Riemannian distances are encoded in the graph's edges.
    """

    def __init__(self, size, sigmas, K, path_to_graph, kernel="gaussian"):
        """
        Args:
            size (list of ints): size of the spherical grid in format (ns, nalpha).
            sigmas (tuple of floats): anisotropy's coefficients.
            K (int): number of neighbors.
            path_to_graph (str): path to the folder to save graph.
        """

        self.manifold = "so3"

        if len(size) != 2:
            raise ValueError(f"size must be 2-dimensional")

        if len(sigmas) != 3:
            raise ValueError(f"sigmas must be 3-dimensional")

        super().__init__(size, sigmas, K, path_to_graph, kernel)

    def uniform_sampling(self, size):
        return so3_uniform_sampling(*size)

    @property
    def centroid_vertex(self):
        return 0

    def riemannian_sqdist(self, Gg, Gh, Re):
        """
        Return the riemannian squared distance between GL(3) group elements Gg and Gh according to the
        riemannian metric Re.

        Args:
            Gg (`torch.FloatTensor`): GL(3) group elements.
            Gh (`torch.FloatTensor`): GL(3) group elements.
            Re (`torch.FloatTensor`): riemannian metric.

        Returns:
            (`torch.FloatTensor`): squared riemannian distance.
        """
        return so3_riemannian_sqdist(Gg, Gh, Re)

    @property
    def group_element(self):
        """
        Return the group elements of graph's vertices.

        Returns:
            (`torch.FloatTensor`): SO(3) group's elements.
        """
        return self.vertex_alpha, self.vertex_beta, self.vertex_gamma

    @property
    def general_linear_group_element(self):
        """
        Return the general linear group elements of graph's vertices.

        Returns:
            (`torch.FloatTensor`): GL(3) group's elements.
        """
        return so3_matrix(self.vertex_alpha, self.vertex_beta, self.vertex_gamma)

    @property
    def group_dim(self):
        """
        Return the name of the group's dimensions.

        Returns:
            (dict): mapping from dimensions' names to dimensions' indices.
        """
        return ["alpha", "beta", "gamma"]

    @property
    def dim(self):
        return self.size[1], self.size[0]

    def cartesian_pos(self, axis=None):
        """
        Return the cartesian positions of the graph's vertices.

        Args:
            axis (str, optional): cartesian axis. If None, return all axis. Defaults to None.

        Returns:
            (`torch.FloatTensor`, optional): x positions.
            (`torch.FloatTensor`, optional): y positions.
            (`torch.FloatTensor`, optional): z positions.
        """

        if not axis is None:
            return (2 * math.pi + self.vertex_alpha) * betagamma2xyz(self.vertex_beta, self.vertex_gamma, axis)

        x, y, z = betagamma2xyz(self.vertex_beta, self.vertex_gamma, axis)
        x *= 2 * math.pi + self.vertex_alpha
        y *= 2 * math.pi + self.vertex_alpha
        z *= 2 * math.pi + self.vertex_alpha

        return x, y, z

    @property
    def vertex_attributes(self):
        """
        Returns the graph's vertices attributes.

        Returns:
            (tuple): tuple of vertices' attributes
        """
        return ("vertex_alpha", "vertex_beta", "vertex_gamma")


class S2GEGraph(GEGraph):
    """
    Approximation of the S(2) manifold. The vertices correspond to manifold elements and the anisotropic
    Riemannian distances are encoded in the graph's edges.
    """

    def __init__(self, size, sigmas, K, path_to_graph, kernel="gaussian"):
        """
        Args:
            size (list of ints): size of the spherical grid in format (ns, nalpha).
            sigmas (tuple of floats): anisotropy's coefficients.
            K (int): number of neighbors.
            path_to_graph (str): path to the folder to save graph.
        """

        self.manifold = "s2"

        if len(size) != 2:
            raise ValueError(f"size must be 2-dimensional")

        if len(sigmas) != 3:
            raise ValueError(f"sigmas must be 2-dimensional")

        super().__init__(size, sigmas, K, path_to_graph, kernel)

    def uniform_sampling(self, size):
        return s2_uniform_sampling(size[0])

    @property
    def centroid_vertex(self):
        return 0

    def riemannian_sqdist(self, Gg, Gh, Re):
        """
        Return the riemannian squared distance between GL(3) group elements Gg and Gh according to the
        riemannian metric Re.

        Args:
            Gg (`torch.FloatTensor`): GL(3) group elements.
            Gh (`torch.FloatTensor`): GL(3) group elements.
            Re (`torch.FloatTensor`): riemannian metric.

        Returns:
            (`torch.FloatTensor`): squared riemannian distance.
        """
        return s2_riemannian_sqdist(Gg, Gh, Re)

    @property
    def group_element(self):
        """
        Return the group elements of graph's vertices.

        Returns:
            (`torch.FloatTensor`): SO(3) group's elements.
        """
        return self.vertex_beta, self.vertex_gamma

    @property
    def general_linear_group_element(self):
        """
        Return the general linear group elements of graph's vertices.

        Returns:
            (`torch.FloatTensor`): GL(3) group's elements.
        """
        return s2_matrix(self.vertex_beta, self.vertex_gamma)

    @property
    def group_dim(self):
        """
        Return the name of the group's dimensions.

        Returns:
            (dict): mapping from dimensions' names to dimensions' indices.
        """
        return ["beta", "gamma"]

    @property
    def dim(self):
        return 1, self.size[0]

    def cartesian_pos(self, axis=None):
        """
        Return the cartesian positions of the graph's vertices.

        Args:
            axis (str, optional): cartesian axis. If None, return all axis. Defaults to None.

        Returns:
            (`torch.FloatTensor`, optional): x positions.
            (`torch.FloatTensor`, optional): y positions.
            (`torch.FloatTensor`, optional): z positions.
        """
        return betagamma2xyz(self.vertex_beta, self.vertex_gamma, axis)

    @property
    def vertex_attributes(self):
        """
        Returns the graph's vertices attributes.

        Returns:
            (tuple): tuple of vertices' attributes
        """
        return ("vertex_beta", "vertex_gamma")
