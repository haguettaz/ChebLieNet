import torch
from torch_geometric.data import Batch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.max_pool import _max_pool_x
from torch_geometric.nn.pool.pool import pool_batch, pool_pos
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_mean
from torch_sparse import coalesce





def max_pool(cluster, data, edge_red="add", transform=None):
    """
    Pools and coarsens a graph given by the Data object according to the clustering
    defined in cluster.
    All nodes within the same cluster will be represented as one node.
    Final node features are defined by the maximum features of all nodes
    within the same cluster, node positions are averaged and edge indices are
    defined to be the union of the edge indices of all nodes within the same
    cluster with the given edge pool operation to reduce the edges' attributes.

    Args:
        cluster (LongTensor): the cluster vector which assigns each node to a specific cluster.
        data (Data): the graph Data object.
        edge_red (str, optional): the operation to reduce edges attributes. It takes value in
            "add", "mean" or "max". Defaults to "add".
        transform (callable, optional): the function/transform that takes in the coarsened and pooled
            Data object and returns a transformed version. Defaults to None.

    Returns:
        (Data): the pooled/coarsened graph Data object.
    """
    cluster, perm = consecutive_cluster(cluster)

    index, attr = pool_edge(cluster, data.edge_index, data.edge_attr, edge_red)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos)

    if transform is not None:
        data = transform(data)

    return data


def pool_edge(cluster, edge_index, edge_attr=None, edge_red="add"):
    """
    Reduce edges based on the clustering allocation defined in the cluster vector, according to
    the given edge reduction operation.

    Args:
        cluster (torch.tensor): the cluster vector which assigns each node to a specific cluster.
        edge_index (torch.tensor): the tensor of edge index of the original graph.
        edge_attr (torch.tensor, optional): the tensor of edge attributes of the original graph.
            Defaults to None.
        edge_red (str, optional): the edge's reduction's operation. Defaults to "add".

    Returns:
        (torch.tensor): the tensor of edge index of the coarsened graph.
        (torch.tensor): the tensor of edge attributes of the coarsened graph.
    """
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes, op=edge_red)
    return edge_index, edge_attr


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
