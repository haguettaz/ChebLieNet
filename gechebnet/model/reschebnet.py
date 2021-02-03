from typing import Optional, Tuple

import torch
from torch import FloatTensor
from torch import device as Device
from torch.nn import AvgPool1d, BatchNorm1d, Identity, LogSoftmax, MaxPool1d, Module, ReLU
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..utils import sparse_tensor_diag
from .convolution import ChebConv


class ResidualBlock(Module):
    """
    Base class for residual blocks.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, K: int):
        """
        Inits the residual block with 2 chebyschev convolutional layers, relu activation function
        and batch normalization.

        Args:
            channels (int): channels.
            K (int): Chebschev's polynomials' order.
        """
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.relu1 = ReLU()
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, out_channels, K)
        self.relu2 = ReLU()

        if in_channels == out_channels:
            self.resizer = Identity()
        else:
            self.resizer = ChebConv(in_channels, out_channels, 1)

    def forward(self, x: FloatTensor, laplacian: SparseFloatTensor) -> FloatTensor:
        """
        Forward pass.

        Args:
            x (FloatTensor): input.
            laplacian (SparseFloatTensor): graph laplacian.

        Returns:
            FloatTensor: output.
        """
        out = self.conv1(x, laplacian)  # (B, C, V)
        out = self.relu1(out)  # (B, C, V)
        out = self.bn2(out)  # (B, C, V)
        out = self.conv2(out, laplacian)  # (B, C, V)
        return self.relu2(out + self.resizer(x))  # (B, C, V)


class ResGEChebNet(Module):
    """
    Residual group equivariant ChebNet.
    """

    def __init__(
        self,
        graph: Graph,
        K: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        pooling: Optional[str] = "max",
        device: Optional[Device] = None,
    ):
        """
        Inits a residual group equivariant chebnet 3 residual's blocks, batch normalization, global pooling
        and predictive logsoftmax activation function.

        Args:
            graph (Graph): graph.
            K (int): degree of the Chebyschev polynomials, the sum goes from indices 0 to K-1.
            in_channels (int): number of dimensions of the input layer.
            hidden_channels (int): number of dimensions of the hidden layers.
            out_channels (int): number of dimensions of the output layer.
            pooling (str, optional): global pooling function. Defaults to 'max'.
            device (Device, optional): computation device. Defaults to None.

        Raises:
            ValueError: pooling must be 'avg' or 'max'
        """
        super(ResGEChebNet, self).__init__()

        self.laplacian = self._normlaplacian(
            graph.laplacian(device), lmax=2.0, num_nodes=graph.num_nodes
        )

        if pooling not in {"avg", "max"}:
            raise ValueError(
                f"{pooling} is not a valid value for pooling: must be 'avg' or 'max'"
            )

        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.relu1 = ReLU()

        self.bn2 = BatchNorm1d(hidden_channels)
        self.resblock2 = ResidualBlock(hidden_channels, hidden_channels, hidden_channels, K)

        self.bn3 = BatchNorm1d(hidden_channels)
        self.resblock3 = ResidualBlock(hidden_channels, hidden_channels, hidden_channels, K)

        self.bn4 = BatchNorm1d(hidden_channels)
        self.resblock4 = ResidualBlock(hidden_channels, hidden_channels, hidden_channels, K)

        self.bn5 = BatchNorm1d(hidden_channels)
        self.conv5 = ChebConv(hidden_channels, out_channels, K)
        self.relu5 = ReLU()

        if pooling == "avg":
            self.pool = AvgPool1d(graph.num_nodes)  # theoretical equivariance
        else:
            self.pool = MaxPool1d(
                graph.num_nodes
            )  # adds some non linearities, better in practice

        self.logsoftmax = LogSoftmax(dim=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Forward pass.

        Args:
            x (FloatTensor): the batch to feed the network with.

        Returns:
            (FloatTensor): the predictions on the batch.
        """
        # Input layer
        out = self.conv1(x, self.laplacian)  # (B, C, V)
        out = self.relu1(out)

        # Hidden layers
        out = self.bn2(out)
        out = self.resblock2(out, self.laplacian)  # (B, C, V)
        out = self.bn3(out)
        out = self.resblock3(out, self.laplacian)  # (B, C, V)
        out = self.bn4(out)
        out = self.resblock4(out, self.laplacian)  # (B, C, V)

        # Output layer
        out = self.bn5(out)
        out = self.conv5(out, self.laplacian)
        out = self.relu5(out)
        out = self.pool(out).squeeze()  # (B, C)
        return self.logsoftmax(out)  # (B, C)

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
