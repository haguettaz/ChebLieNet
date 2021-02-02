from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor
from torch import device as Device
from torch.nn import AvgPool1d, BatchNorm1d, MaxPool1d, Module, ReLU, Softmax
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..utils import sparse_tensor_diag
from .convolution import ChebConv


class ResidualBlock(Module):
    """
    Base class for residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, K: int):
        """
        Inits the residual block with 2 chebyschev convolutional layers, relu activation function
        and batch normalization.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            hidden_channels (int): hidden channels.
            K (int): Chebschev's polynomials' order.
        """
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.relu1 = ReLU()
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, out_channels, K)
        self.relu2 = ReLU()

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
        return self.relu2(out + x)  # (B, C, V)


class ResGEChebNet(Module):
    """
    Residual group equivariant ChebNet.
    """

    def __init__(
        self,
        graph: Graph,
        K: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 20,
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
            out_channels (int): number of dimensions of the output layer.
            hidden_channels (int, optional): number of dimensions of the hidden layers. Defaults to 20.
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
            raise ValueError(f"{pooling} is not a valid value for pooling: must be 'avg' or 'max'")

        self.resblock1 = ResidualBlock(in_channels, hidden_channels, hidden_channels, K)

        self.bn2 = BatchNorm1d(hidden_channels)
        self.resblock2 = ResidualBlock(in_channels, hidden_channels, hidden_channels, K)

        self.bn3 = BatchNorm1d(hidden_channels)
        self.resblock3 = ResidualBlock(in_channels, hidden_channels, out_channels, K)

        if pooling == "avg":
            self.pool = AvgPool1d(graph.num_nodes)  # theoretical equivariance
        else:
            self.pool = MaxPool1d(graph.num_nodes)  # adds some non linearities, better in practice

        self.logsoftmax = Softmax(dim=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Forward pass.

        Args:
            x (FloatTensor): the batch to feed the network with.

        Returns:
            (FloatTensor): the predictions on the batch.
        """
        # Input layer
        out = self.resblock1(x, self.laplacian)  # (B, C, V)

        # Hidden layers
        out = self.bn2(out)
        out = self.resblock2(out, self.laplacian)  # (B, C, V)
        out = self.bn3(out)
        out = self.resblock3(out, self.laplacian)  # (B, C, V)

        # Output layer
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


class GEChebNet(Module):
    """
    Group equivariant ChebNet.
    """

    def __init__(
        self,
        graph: Graph,
        K: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 20,
        pooling: Optional[str] = "max",
        device: Optional[Device] = None,
    ):
        """
        Inits a chebnet with 4 convolutional layers, relu activation functions, batch normalization global pooling
        and predictive logsoftmax activation function.

        Args:
            graph (Graph): graph.
            K (int): degree of the Chebyschev polynomials, the sum goes from indices 0 to K-1.
            in_channels (int): number of dimensions of the input layer.
            out_channels (int): number of dimensions of the output layer.
            hidden_channels (int, optional): number of dimensions of the hidden layers. Defaults to 20.
            pooling (str, optional): global pooling function. Defaults to 'max'.
            device (Device, optional): computation device. Defaults to None.

        Raises:
            ValueError: pooling must be 'avg' or 'max'
        """
        super(GEChebNet, self).__init__()

        self.laplacian = self._normlaplacian(
            graph.laplacian(device), lmax=2.0, num_nodes=graph.num_nodes
        )

        if pooling not in {"avg", "max"}:
            raise ValueError(f"{pooling} is not a valid value for pooling: must be 'avg' or 'max'")

        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.relu1 = ReLU()

        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.relu2 = ReLU()
        self.bn3 = BatchNorm1d(hidden_channels)
        self.conv3 = ChebConv(hidden_channels, hidden_channels, K)
        self.relu3 = ReLU()
        self.bn4 = BatchNorm1d(hidden_channels)
        self.conv4 = ChebConv(hidden_channels, out_channels, K)
        self.relu4 = ReLU()

        if pooling == "avg":
            self.pool = AvgPool1d(graph.num_nodes)  # theoretical equivariance
        else:
            self.pool = MaxPool1d(graph.num_nodes)  # adds some non linearities, better in practice

        self.logsoftmax = Softmax(dim=1)

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
        out = self.relu1(out)  # (B, C, V)

        # Hidden layers
        out = self.bn2(out)  # (B, C, V)
        out = self.conv2(out, self.laplacian)  # (B, C, V)
        out = self.relu2(out)  # (B, C, V)
        out = self.bn3(out)  # (B, C, V)
        out = self.conv3(out, self.laplacian)  # (B, C, V)
        out = self.relu3(out)  # (B, C, V)
        out = self.bn4(out)  # (B, C, V)
        out = self.conv4(out, self.laplacian)  # (B, C, V)
        out = self.relu4(out)  # (B, C, V)

        # Output layer
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
