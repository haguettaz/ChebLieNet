from typing import Optional, Tuple, Union

import torch
from torch import FloatTensor
from torch import device as Device
from torch.nn import AvgPool1d, BatchNorm1d, Linear, LogSoftmax, MaxPool1d, Module, ModuleList, ReLU
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..graph.signal_processing import get_norm_laplacian
from ..graph.sparsification import sparsify_on_edges, sparsify_on_nodes
from .convolution import ChebConv


class GEChebNet(Module):
    """
    Group equivariant ChebNet.
    """

    def __init__(
        self,
        graph: Graph,
        K: int,
        in_channels: int,
        hidden_channels: list,
        out_channels: int,
        pooling: Optional[str] = "max",
        device: Optional[Device] = None,
    ):
        """
        Inits a chebnet with 4 convolutional layers, relu activation functions, batch normalization global pooling
        and predictive logsoftmax activation function.

        Args:
            laplacian (SparseFloatTensor): symmetric normalized graph laplacian.
            K (int): degree of the Chebyschev polynomials, the sum goes from indices 0 to K-1.
            in_channels (int): number of dimensions of the input layer.
            hidden_channels (int, optional): number of dimensions of the hidden layers. Defaults to 20.
            out_channels (int): number of dimensions of the output layer.
            pooling (str, optional): global pooling function. Defaults to 'max'.
            device (Device, optional): computation device. Defaults to None.

        Raises:
            ValueError: pooling must be 'avg' or 'max'
        """
        super(GEChebNet, self).__init__()

        self.device = device

        if pooling not in {"avg", "max"}:
            raise ValueError(f"{pooling} is not a valid value for pooling: must be 'avg' or 'max'!")

        self.device = device
        self.graph = graph  # laplacian is stored on cpu

        self.norm_laplacian = get_norm_laplacian(
            graph.edge_index, graph.edge_weight, graph.num_nodes, 2.0, self.device
        )

        # self.node_index = graph.node_index

        self.in_conv = ChebConv(in_channels, hidden_channels[0], K)
        self.in_relu = ReLU()

        self.hidden_layers = len(hidden_channels) - 1

        if self.hidden_layers:
            self.hidden_bn = ModuleList(
                [BatchNorm1d(hidden_channels[l]) for l in range(self.hidden_layers)]
            )
            self.hidden_conv = ModuleList(
                [
                    ChebConv(hidden_channels[l], hidden_channels[l + 1], K)
                    for l in range(self.hidden_layers)
                ]
            )
            self.hidden_relu = ModuleList([ReLU()] * self.hidden_layers)

        # theoretical equivariance
        if pooling == "avg":
            self.global_pooling = AvgPool1d(graph.num_nodes)
        # adds some non linearities, better in practice
        else:
            self.global_pooling = MaxPool1d(graph.num_nodes)

        self.out_bn = BatchNorm1d(hidden_channels[-1])
        self.out_lin = Linear(hidden_channels[-1], out_channels)
        # self.out_conv = ChebConv(hidden_channels[-1], out_channels, K)
        self.out_relu = ReLU()

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
        out = self.in_conv(x, self.norm_laplacian)  # (B, C, V)
        out = self.in_relu(out)  # (B, C, V)

        # Hidden layers
        for l in range(self.hidden_layers):
            out = self.hidden_bn[l](out)  # (B, C, V)
            out = self.hidden_conv[l](out, self.norm_laplacian)  # (B, C, V)
            out = self.hidden_relu[l](out)  # (B, C, V)

        # Output layer
        out = self.global_pooling(out).squeeze()  # (B, C)
        out = self.out_bn(out)  # (B, C)
        # out = self.out_conv(out, self.norm_laplacian)
        out = self.out_lin(out)  # (B, C)
        out = self.out_relu(out)
        return self.logsoftmax(out)  # (B, C)

    def sparsify_laplacian(self, on, rate):
        if rate == 0.0:
            self.norm_laplacian = get_norm_laplacian(
                self.graph.edge_index,
                self.graph.edge_weight,
                self.graph.num_nodes,
                2.0,
                self.device,
            )
            # self.node_index = self.graph.node_index
            return

        if on == "edges":
            edge_index, edge_weight = sparsify_on_edges(
                self.graph.edge_index, self.graph.edge_weight, rate
            )
        else:
            edge_index, edge_weight = sparsify_on_nodes(
                self.graph.edge_index,
                self.graph.edge_weight,
                self.graph.node_index,
                self.graph.num_nodes,
                rate,
            )

        # self.node_index = edge_index[0].unique()
        self.norm_laplacian = get_norm_laplacian(
            edge_index, edge_weight, self.graph.num_nodes, 2.0, self.device
        )

    @property
    def capacity(self) -> int:
        """
        Return the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): number of trainable parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())
