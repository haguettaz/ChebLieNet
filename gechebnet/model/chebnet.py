from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn import BatchNorm1d, Module

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
        laplacian_device: Optional[torch.device] = None,
    ):
        """
        Initialize a ChebNet with 6 convolutional layers and batch normalization.

        Args:
            graph (Graph): graph.
            K (int): the degree of the Chebyschev polynomials, the sum goes from indices 0 to K-1.
            in_channels (int): the number of dimensions of the input layer.
            out_channels (int): the number of dimensions of the output layer.
            hidden_channels (int): the number of dimensions of the hidden layers.
            laplacian_device (torch.device, optional): computation device.
        """
        super(GEChebNet, self).__init__()

        laplacian_device = laplacian_device or torch.device("cpu")

        self.conv1 = ChebConv(
            graph, in_channels, hidden_channels, K, laplacian_device=laplacian_device
        )
        self.conv2 = ChebConv(
            graph, hidden_channels, hidden_channels, K, laplacian_device=laplacian_device
        )
        self.conv3 = ChebConv(
            graph, hidden_channels, out_channels, K, laplacian_device=laplacian_device
        )

        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)

        self.nx1, self.nx2, self.nx3 = graph.nx1, graph.nx2, graph.nx3

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

        # Global pooling
        x = torch.mean(x, dim=2)  # (B, C)

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
