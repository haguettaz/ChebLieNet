import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, global_max_pool, max_pool, voxel_grid

from .pooling import orientation_subsampling, spatial_subsampling


class ChebNet(torch.nn.Module):
    def __init__(self, K, num_layers, input_dim=1, output_dim=10, hidden_dim=10):
        """
        Initialize a ChebNet with 6 convolutional layers and batch normalization.

        Args:
            K (int): the degree of the Chebyschev polynomials.
            num_layers (int): the number of layers on the orientation axis.
            input_dim (int, optional): the number of dimensions of the input layer. Defaults to 1.
            output_dim (int, optional): the number of dimensions of the output layer. Defaults to 10.
            hidden_dim (int, optional): the number of dimensions of the hidden layers. Defaults to 10.
        """
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim, K)  # input_dim*hidden_dim*K weights + hidden_dim bias
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K)  # hidden_dim*hidden_dim*K weights + hidden_dim bias

        self.conv3 = ChebConv(hidden_dim, hidden_dim, K)  # hidden_dim*hidden_dim*K weights + hidden_dim bias
        self.conv4 = ChebConv(hidden_dim, hidden_dim, K)  # hidden_dim*hidden_dim*K weights + hidden_dim bias

        self.conv5 = ChebConv(hidden_dim, hidden_dim, K)  # hidden_dim*hidden_dim*K weights + hidden_dim bias
        self.conv6 = ChebConv(hidden_dim, output_dim, K)  # hidden_dim*output_dim*K weights + output_dim bias

        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim)

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
