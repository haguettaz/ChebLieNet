import math
from typing import Optional, Tuple, Union

import torch
from torch import FloatTensor
from torch import device as Device
from torch.nn import AdaptiveMaxPool1d, BatchNorm1d, Linear, LogSoftmax, Module, ModuleList, ReLU
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..graph.signal_processing import get_norm_laplacian
from ..graph.sparsification import sparsify_on_edges, sparsify_on_nodes
from .convolution import ChebConv
from .utils import BasicBlock, NetworkBlock


class WideGEChebNet(Module):
    def __init__(self, in_channels, out_channels, K, graph, depth, widen_factor=1):
        super(WideGEChebNet, self).__init__()

        self.graph = graph

        hidden_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        if (depth - 2) % 3:
            raise ValueError(f"{depth} is not a valid value for depth")

        num_layers = (depth - 2) // 3

        # 1st conv before any network block
        self.conv = ChebConv(in_channels, hidden_channels[0], K)

        # 1st block
        self.block1 = NetworkBlock(
            hidden_channels[0], hidden_channels[1], num_layers, BasicBlock, K
        )
        # 2nd block
        self.block2 = NetworkBlock(
            hidden_channels[1], hidden_channels[2], num_layers, BasicBlock, K
        )
        # 3rd block
        self.block3 = NetworkBlock(
            hidden_channels[2], hidden_channels[3], num_layers, BasicBlock, K
        )

        # global average pooling and classifier
        self.globalmaxpool = AdaptiveMaxPool1d(1)
        self.fc = Linear(hidden_channels[3], out_channels)
        self.logsoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        laplacian = self.graph.laplacian.to(x.device)
        B, C, _ = x.shape

        out = self.conv(x, laplacian)

        out = self.block1(out, laplacian)
        out = self.block2(out, laplacian)
        out = self.block3(out, laplacian)

        out = self.globalmaxpool(out).contiguous().view(B, C)
        out = self.fc(out)
        out = self.logsoftmax(out)

        return out

    @property
    def capacity(self) -> int:
        """
        Return the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): number of trainable parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())


def wide_gechebnet_5_2(in_channels, out_channels, K, graph):
    return WideGEChebNet(in_channels, out_channels, K, graph, depth=5, widen_factor=2)


def wide_gechebnet_8_2(in_channels, out_channels, K, graph):
    return WideGEChebNet(in_channels, out_channels, K, graph, depth=8, widen_factor=2)


def wide_gechebnet_8_4(in_channels, out_channels, K, graph):
    return WideGEChebNet(in_channels, out_channels, K, graph, depth=8, widen_factor=4)
