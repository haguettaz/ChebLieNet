import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, global_max_pool, max_pool, voxel_grid

from .pooling import orientation_subsampling, spatial_subsampling


class ChebNet(torch.nn.Module):
    def __init__(self, K, hidden_dim, num_layers):
        """
        Initialize a ChebNet with 6 convolutional layers and batch normalization.

        Args:
            K (int): the degree of the Chebyschev polynomials.
            hidden_dim (int): the number of dimensions of the hidden layers.
            num_layers (int): the number of layers on the orientation axis.
        """
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(1, hidden_dim, K)  # 1*16*K weights + 16 bias
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K)  # 16*32*K weights + 32 bias

        self.conv3 = ChebConv(hidden_dim, hidden_dim, K)  # 32*32*K weights + 32 bias
        self.conv4 = ChebConv(hidden_dim, hidden_dim, K)  # 32*10*K weights + 10 bias

        self.conv5 = ChebConv(hidden_dim, hidden_dim, K)  # 32*32*K weights + 32 bias
        self.conv6 = ChebConv(hidden_dim, hidden_dim, K)  # 32*10*K weights + 10 bias

        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(16)
        self.bn4 = torch.nn.BatchNorm1d(16)
        self.bn5 = torch.nn.BatchNorm1d(16)

        self.num_layers = num_layers

    def forward(self, data):
        """
        Forward function receiving as input a batch and outputing a prediction on this batch

        Args:
            data (Batch): the batch to feed the network with.

        Returns:
            (torch.tensor): the predictions on the batch.
        """
        data.x = self.conv1(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn1(data.x)
        data.x = data.x.relu()

        data.x = self.conv2(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn2(data.x)
        data.x = data.x.relu()

        cluster = spatial_subsampling(data.pos, data.batch, 2.0)
        data = max_pool(cluster, data)

        data.x = self.conv3(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn3(data.x)
        data.x = data.x.relu()

        data.x = self.conv4(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn4(data.x)
        data.x = data.x.relu()

        cluster = spatial_subsampling(data.pos, data.batch, 2.0)
        data = max_pool(cluster, data)

        data.x = self.conv5(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = self.bn5(data.x)
        data.x = data.x.relu()

        data.x = self.conv6(data.x, data.edge_index, data.edge_attr, data.batch)
        data.x = data.x.relu()

        cluster = spatial_subsampling(data.pos, data.batch, 2.0)
        data = max_pool(cluster, data)

        cluster = orientation_subsampling(data.pos, data.batch, float(self.num_layers))
        data = max_pool(cluster, data)

        data.x = global_max_pool(data.x, data.batch)

        return F.log_softmax(data.x, dim=1)

    @property
    def capacity(self):
        """
        Return the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): the number of parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())
