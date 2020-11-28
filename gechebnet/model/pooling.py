import torch
from torch_geometric.nn.pool import voxel_grid


def spatial_subsampling(pos, batch, divider=2.0):
    """
    Performs spatial subsampling, based on node's positions.

    Args:
        pos (torch.tensor): the node's positions.
        batch (torch.tensor): the node's batches
        divider (float, optional): the spatial divider. Defaults to 2.0.

    Returns:
        (torch.tensor): the clustering resulting of the spatial subsampling.
    """
    return voxel_grid(pos, batch, torch.tensor([divider, divider, 1.0]))


def orientation_subsampling(pos, batch, divider=2.0):
    """
    Performs orientation subsampling, based on node's positions.

    Args:
        pos (torch.tensor): the node's positions.
        batch (torch.tensor): the node's batches
        divider (float, optional): the orientation divider. Defaults to 2.0.

    Returns:
        (torch.tensor): the clustering resulting of the orientation subsampling.
    """
    return voxel_grid(pos, batch, torch.tensor([1.0, 1.0, divider]))
