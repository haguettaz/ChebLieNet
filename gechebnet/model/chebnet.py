from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor
from torch import device as Device
from torch.nn import AvgPool1d, BatchNorm1d, MaxPool1d, Module
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..utils import sparse_tensor_diag
from .convolution import ChebConv


class GEChebNet(Module):
    def __init__(
        self,
        graph: Graph,
        K: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 20,
        hidden_layers: int = 2,
        pooling: str = "max",
        device: Device = None,
    ):
        """
        Initialize a ChebNet with convolutional layers and batch normalization.

        Args:
            graph (Graph): graph.
            K (int): degree of the Chebyschev polynomials, the sum goes from indices 0 to K-1.
            in_channels (int): number of dimensions of the input layer.
            out_channels (int): number of dimensions of the output layer.
            hidden_channels (int, optional): number of dimensions of the hidden layers. Defaults to 20.
            hidden_layers (int, optional): number of hidden layers. Defaults to 2.
            pooling (str, optional): global pooling function. Defaults to 'max'.

        Raises:
            ValueError: pooling must be 'avg' or 'max'
        """
        super(GEChebNet, self).__init__()

        self.laplacian = self._normlaplacian(
            graph.laplacian(device), lmax=2.0, num_nodes=graph.num_nodes
        )
        self.hidden_layers = hidden_layers

        if pooling not in {"avg", "max"}:
            raise ValueError(f"{pooling} is not a valid value for pooling: must be 'avg' or 'max'")

        self.in_conv = ChebConv(in_channels, hidden_channels, K)

        self.hidden_bn = torch.nn.ModuleList([BatchNorm1d(hidden_channels)] * hidden_layers)
        self.hidden_conv = torch.nn.ModuleList(
            [ChebConv(hidden_channels, hidden_channels, K)] * hidden_layers
        )

        self.out_bn = BatchNorm1d(hidden_channels)
        self.out_conv = ChebConv(hidden_channels, out_channels, K)

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
        # Input layer
        x = self.in_conv(x, self.laplacian)  # (B, C, V)
        x = F.relu(x)  # (B, C, V)

        # Hidden layers
        for l in range(self.hidden_layers):
            x = self.hidden_bn[l](x)  # (B, C, V)
            x = self.hidden_conv[l](x, self.laplacian)  # (B, C, V)
            x = F.relu(x)  # (B, C, V)

        # Output layer
        x = self.out_bn(x)  # (B, C, V)
        x = self.out_conv(x, self.laplacian)  # (B, C, V)
        x = F.relu(x)  # (B, C, V)
        x = self.pool(x).squeeze()  # (B, C)
        x = F.log_softmax(x, dim=1)  # (B, C)

        return x

    def _normlaplacian(
        self, laplacian: SparseFloatTensor, lmax: float, num_nodes: int
    ) -> SparseFloatTensor:
        """Scale the eigenvalues from [0, lmax] to [-1, 1]."""
        return 2 * laplacian / lmax - sparse_tensor_diag(num_nodes, device=laplacian.device)

    @property
    def capacity(self) -> int:
        """
        Return the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): number of trainable parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())
