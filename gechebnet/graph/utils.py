import math

import torch


def metric_tensor(theta, sigmas):
    """
    Return the anisotropic metric tensor, the main directions of the kernel are:
        1. aligned with theta and orthogonal to the orientation axis.
        2. orthogonal to theta and to the orientation axis.
        3. aligned with the orientation axis.

    Args:
        theta (float): the orientation of the first main direction of the kernel (in radians).
        sigmas (tuple): the intensities of the three main anisotropic directions.

    Returns:
        (torch.tensor): the metric tensor with shape (3, 3).
    """

    sigma_1, sigma_2, sigma_3 = sigmas
    e1 = torch.tensor([math.cos(theta), math.sin(theta), 0], dtype=torch.float32)
    e2 = torch.tensor([-math.sin(theta), math.cos(theta), 0], dtype=torch.float32)
    e3 = torch.tensor([0, 0, 1], dtype=torch.float32)

    D = e1.unsqueeze(1) * e1.unsqueeze(0) * sigma_1
    D += e2.unsqueeze(1) * e2.unsqueeze(0) * sigma_2
    D += e3.unsqueeze(1) * e3.unsqueeze(0) * sigma_3

    return D


class WeightKernel:
    """
    The base class for the weight kernels.
    """

    def __init__(self, threshold=0.5, sigma=1.0, *args, **kwargs):
        """
        Initialize the weight kernel with hyperparameters.

        Args:
            threshold (float): the threshold on the weights.
            sigma (float): the sigma parameter of the kernel.
        """
        self.threshold = threshold
        self.sigma = sigma

    def compute(self, sq_dist):
        """
        Compute the edge weights from the distances between each pair of nodes. All nodes with distances below
        a threshold have an equal weighted edges between them.

        Args:
            sq_dist (torch.tensor): the tensor of pairwise squared distances.

        Returns:
            (torch.tensor): the tensor of edge's weights
        """
        mask_threshold = sq_dist <= self.threshold
        weights = torch.zeros(sq_dist.shape)
        weights[mask_threshold] = self.sigma
        return weights


class GaussianKernel(WeightKernel):
    def compute(self, sq_dist):
        """
        Compute the edge weights from the distances between each pair of nodes using a gaussian kernel.

        Args:
            sq_dist (torch.tensor): the tensor of pairwise squared distances.

        Returns:
            (torch.tensor): the tensor of edge's weights
        """
        weights = torch.exp(-(sq_dist ** 2) / (2 * self.sigma ** 2))
        weights[weights < self.threshold] = 0.0
        return weights


class CauchyKernel(WeightKernel):
    def compute(self, sq_dist):
        """
        Compute the edge weights from the distances between each pair of nodes using a cauchy kernel.

        Args:
            sq_dist (torch.tensor): the tensor of pairwise squared distances.

        Returns:
            (torch.tensor): the tensor of edge's weights
        """
        weights = torch.div(1, 1 + torch.div(sq_dist, self.sigma ** 2))
        weights[weights < self.threshold] = 0.0
        return weights
