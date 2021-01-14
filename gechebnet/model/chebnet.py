from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn import AvgPool1d, BatchNorm1d, MaxPool1d, Module

from ..graph.graph import Graph
from .convolution import ChebConv


class GEChebNet(Module):
    def __init__(
        self,
        graph: Graph,
        K: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        pooling: str,
        laplacian_device: Optional[torch.device] = None,
    ):
        """
        Initialize a ChebNet with 6 convolutional layers and batch normalization.

        Args:
            graph (Graph): graph.
            K (int): degree of the Chebyschev polynomials, the sum goes from indices 0 to K-1.
            in_channels (int): number of dimensions of the input layer.
            out_channels (int): number of dimensions of the output layer.
            hidden_channels (int): number of dimensions of the hidden layers.
            pooling (str): global pooling function
            laplacian_device (torch.device, optional): computation device. Defaults to None.

        Raises:
            ValueError: pooling must be 'avg' or 'max'
        """
        super(GEChebNet, self).__init__()

        laplacian_device = laplacian_device or torch.device("cpu")

        if pooling not in {"avg", "max"}:
            raise ValueError(f"{pooling} is not a valid value for pooling: must be 'avg' or 'max'")

        self.conv1 = ChebConv(
            graph, in_channels, hidden_channels, K, laplacian_device=laplacian_device
        )
        self.conv2 = ChebConv(
            graph, hidden_channels, hidden_channels, K, laplacian_device=laplacian_device
        )
        self.conv3 = ChebConv(
            graph, hidden_channels, hidden_channels, K, laplacian_device=laplacian_device
        )
        self.conv4 = ChebConv(
            graph, hidden_channels, out_channels, K, laplacian_device=laplacian_device
        )

        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)

        self.nx1, self.nx2, self.nx3 = graph.nx1, graph.nx2, graph.nx3

        if pooling == "avg":
            self.pool = AvgPool1d(graph.num_nodes)  # theoretical equivariance
        else:
            self.pool = MaxPool1d(graph.num_nodes)  # adds some non linearities, better in practice

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Forward function receiving as input a batch and outputing a prediction on this batch

        Args:
            x (FloatTensor): the batch to feed the network with.

        Returns:
            (FloatTensor): the predictions on the batch.
        """
        # Chebyschev Convolutions
        x = self.conv1(x)  # (B, C, V)
        x = F.relu(x)  # (B, C, V)

        x = self.bn2(x)  # (B, C, V)
        x = self.conv2(x)  # (B, C, V)
        x = F.relu(x)  # (B, C, V)

        x = self.bn3(x)  # (B, C, V)
        x = self.conv3(x)  # (B, C, V)
        x = F.relu(x)  # (B, C, V)

        x = self.bn4(x)  # (B, C, V)
        x = self.conv4(x)  # (B, C, V)
        x = F.relu(x)  # (B, C, V)

        # Global pooling
        x = self.pool(x).squeeze()  # (B, C)

        # Output layer
        x = F.log_softmax(x, dim=1)  # (B, C)

        return x

    @property
    def capacity(self) -> int:
        """
        Return the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): number of trainable parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())
