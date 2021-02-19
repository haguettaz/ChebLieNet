import math
from typing import Optional, Tuple, Union

import torch
from torch import FloatTensor, nn
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..graph.signal_processing import get_norm_laplacian
from ..graph.sparsification import sparsify_on_edges, sparsify_on_nodes
from .convolution import ChebConv
from .utils import BasicBlock, GraphPooling, NetworkBlock


class WideGEChebNet(nn.Module):
    def __init__(self, graph_lvl1, graph_lvl2, graph_lvl3, in_channels, out_channels, R, depth, widen_factor=1):
        super(WideGEChebNet, self).__init__()

        hidden_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        if (depth - 2) % 3:
            raise ValueError(f"{depth} is not a valid value for depth")

        num_layers = (depth - 2) // 3

        # input layer : convolutional layer + relu
        self.conv = ChebConv(graph_lvl1, in_channels, hidden_channels[0], R)
        self.relu = nn.ReLU(inplace=True)

        # hidden layers : 3 convolutional blocks + spatial pooling
        self.block1 = NetworkBlock(graph_lvl1, hidden_channels[0], hidden_channels[1], num_layers, BasicBlock, R)

        if graph_lvl1.nx3 != graph_lvl2.nx3 or graph_lvl1.nx2 != graph_lvl2.nx2 or graph_lvl1.nx1 != graph_lvl2.nx1:
            self.graphmaxpool1 = GraphPooling(
                graph_lvl1.nx3, graph_lvl2.nx3, graph_lvl1.nx2, graph_lvl2.nx2, graph_lvl1.nx1, graph_lvl2.nx1
            )
        else:
            self.graphmaxpool1 = None

        self.block2 = NetworkBlock(graph_lvl2, hidden_channels[1], hidden_channels[2], num_layers, BasicBlock, R)

        if graph_lvl2.nx3 != graph_lvl3.nx3 or graph_lvl2.nx2 != graph_lvl3.nx2 or graph_lvl2.nx1 != graph_lvl3.nx1:
            self.graphmaxpool2 = GraphPooling(
                graph_lvl2.nx3, graph_lvl3.nx3, graph_lvl2.nx2, graph_lvl3.nx2, graph_lvl2.nx1, graph_lvl3.nx1
            )
        else:
            self.graphmaxpool2 = None

        self.block3 = NetworkBlock(graph_lvl3, hidden_channels[2], hidden_channels[3], num_layers, BasicBlock, R)

        # output layer : global average pooling + fc
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_channels[3], out_channels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        B, _, _ = x.shape

        out = self.relu(self.conv(x))

        out = self.block1(out)
        if self.graphmaxpool1 is not None:
            out = self.graphmaxpool1(out)
        out = self.block2(out)
        if self.graphmaxpool2 is not None:
            out = self.graphmaxpool2(out)
        out = self.block3(out)

        out = self.globalmaxpool(out).contiguous().view(B, -1)
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


def wide_gechebnet_14_8(graph_lvl1, graph_lvl2, graph_lvl3, in_channels, out_channels, R):
    return WideGEChebNet(graph_lvl1, graph_lvl2, graph_lvl3, in_channels, out_channels, R, depth=14, widen_factor=8)


def wide_gechebnet_11_4(graph_lvl1, graph_lvl2, graph_lvl3, in_channels, out_channels, R):
    return WideGEChebNet(graph_lvl1, graph_lvl2, graph_lvl3, in_channels, out_channels, R, depth=11, widen_factor=4)


def wide_gechebnet_8_2(graph_lvl1, graph_lvl2, graph_lvl3, in_channels, out_channels, R):
    return WideGEChebNet(graph_lvl1, graph_lvl2, graph_lvl3, in_channels, out_channels, R, depth=8, widen_factor=2)
