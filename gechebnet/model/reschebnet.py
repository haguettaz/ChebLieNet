from typing import Optional, Tuple, Union

import torch
from torch import FloatTensor
from torch import device as Device
from torch.nn import (
    AvgPool1d,
    BatchNorm1d,
    Identity,
    LogSoftmax,
    MaxPool1d,
    Module,
    ModuleList,
    ReLU,
)
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..graph.sparsification import get_sparse_laplacian
from ..utils import sparse_tensor_diag
from .convolution import ChebConv


class ResidualBlock(Module):
    """
    Base class for residual blocks.
    """

    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, K: int):
        """
        Inits the residual block with 2 chebyschev convolutional layers, relu activation function
        and batch normalization.

        Args:
            channels (int): channels.
            K (int): Chebschev's polynomials' order.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.K = K

        hidden_in_channels = hidden_channels
        hidden_out_channels = hidden_channels[1:] + [out_channels]

        self.hidden_layers = len(hidden_channels)

        self.in_bn = BatchNorm1d(in_channels)
        self.in_conv = ChebConv(in_channels, hidden_in_channels[0], K)

        self.hidden_relu = ModuleList([ReLU()] * self.hidden_layers)
        self.hidden_bn = ModuleList(
            [BatchNorm1d(hidden_in_channels[l]) for l in range(self.hidden_layers)]
        )
        self.hidden_conv = ModuleList(
            [
                ChebConv(hidden_in_channels[l], hidden_out_channels[l], K)
                for l in range(self.hidden_layers)
            ]
        )

        self.out_relu = ReLU()

        if in_channels == out_channels:
            self.shortcut = Identity()
        else:
            self.shortcut = ChebConv(in_channels, out_channels, 1, bias=False)

    def forward(self, x: FloatTensor, laplacian: SparseFloatTensor) -> FloatTensor:
        """
        Forward pass.

        Args:
            x (FloatTensor): input.
            laplacian (SparseFloatTensor): graph laplacian.

        Returns:
            FloatTensor: output.
        """
        x = self.in_bn(x)
        out = self.in_conv(x, laplacian)  # (B, C, V)

        for l in range(self.hidden_layers):
            out = self.hidden_relu[l](out)
            out = self.hidden_bn[l](out)
            out = self.hidden_conv[l](out, laplacian)

        return self.out_relu(out + self.shortcut(x))  # (B, C, V)

    def extra_repr(self) -> str:
        return (
            "in_channels={in_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}, "
            "K={K}".format(**self.__dict__)
        )


class ResGEChebNet(Module):
    """
    Residual group equivariant ChebNet.
    """

    def __init__(
        self,
        graph: Graph,
        K: int,
        in_channels: int,
        hidden_res_channels: list,
        out_channels: int,
        pooling: Optional[str] = "max",
        sparsification_rate: Optional[float] = None,
        sparsify_on: Optional[str] = None,
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

        self.hidden_res_layers = len(hidden_res_channels)

        if pooling not in {"avg", "max"}:
            raise ValueError(f"{pooling} is not a valid value for pooling: must be 'avg' or 'max'")

        if sparsify_on is not None and sparsify_on not in {"edges", "nodes"}:
            raise ValueError(
                f"{sparsify_on} is not a valid value for sparsify_on: must be 'edges' or 'nodes'!"
            )

        if sparsification_rate is not None and sparsify_on is None:
            raise ValueError(f"sparsify_on must be specified!")

        self.sparsify_on = sparsify_on
        self.sparsification_rate = sparsification_rate or 0.0

        self.in_conv = ChebConv(in_channels, hidden_res_channels[0][0], K)
        self.in_relu = ReLU()

        self.hidden_resblock = ModuleList(
            [
                ResidualBlock(
                    hidden_res_channels[l][0],
                    hidden_res_channels[l][1:-1],
                    hidden_res_channels[l][-1],
                    K,
                )
                for l in range(self.hidden_res_layers)
            ]
        )

        self.out_bn = BatchNorm1d(hidden_res_channels[-1][-1])
        self.out_conv = ChebConv(hidden_res_channels[-1][-1], out_channels, K)
        self.out_relu = ReLU()

        if pooling == "avg":
            self.global_pooling = AvgPool1d(graph.num_nodes)  # theoretical equivariance
        else:
            self.global_pooling = MaxPool1d(
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
        out = self.in_conv(
            x, self.sparse_laplacian if self.sparsification_rate else self.laplacian
        )  # (B, C, V)
        out = self.in_relu(out)

        # Hidden layers
        for l in range(self.hidden_res_layers):
            out = self.hidden_resblock[l](
                out, self.sparse_laplacian if self.sparsification_rate else self.laplacian
            )

        # Output layer
        out = self.out_bn(out)
        out = self.out_conv(
            out, self.sparse_laplacian if self.sparsification_rate else self.laplacian
        )
        out = self.out_relu(out)
        out = self.global_pooling(out).squeeze()  # (B, C)
        return self.logsoftmax(out)  # (B, C)

    def _normlaplacian(
        self, laplacian: SparseFloatTensor, lmax: float, num_nodes: int
    ) -> SparseFloatTensor:
        """Scale the eigenvalues from [0, lmax] to [-1, 1]."""
        return 2 * laplacian / lmax - sparse_tensor_diag(num_nodes, device=laplacian.device)

    def sparsify_laplacian(self):
        self.sparse_laplacian = get_sparse_laplacian(
            self.laplacian, on=self.sparsify_on, sparsification_rate=self.sparsification_rate
        )

    @property
    def capacity(self) -> int:
        """
        Return the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): number of trainable parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())
