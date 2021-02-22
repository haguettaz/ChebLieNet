from typing import Optional

from torch import Tensor, nn

from ..graph.graph import Graph
from .convolution import ChebConv
from .utils import GraphPooling, NetworkBlock, ResidualBlock


class WideResGEChebNet(nn.Module):
    def __init__(
        self,
        graph_lvl1: Graph,
        graph_lvl2: Graph,
        graph_lvl3: Graph,
        in_channels: int,
        out_channels: int,
        R: int,
        depth: int,
        widen_factor: Optional[int] = 1,
    ):
        """
        Initialization.

        Args:
            graph_lvl1 (Graph): graph with level 1 coarsening (original graph).
            graph_lvl2 (Graph): graph with level 2 coarsening.
            graph_lvl3 (Graph): graph with level 3 coarsening.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            R (int): order of the Chebyshev polynomials.
            depth (int): depth of the neural network.
            widen_factor (int, optional): widen factor of the neural network. Defaults to 1.

        Raises:
            ValueError: depth must be compatible with the architecture.
        """
        super(WideResGEChebNet, self).__init__()

        hidden_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        if (depth - 2) % 6:
            raise ValueError(f"{depth} is not a valid value for {depth}")

        num_layers = (depth - 2) // 6

        # input layer : convolutional layer + relu
        self.conv = ChebConv(graph_lvl1, in_channels, hidden_channels[0], R)
        self.relu = nn.ReLU(inplace=True)

        # hidden layers : 3 convolutional blocks
        self.block1 = NetworkBlock(graph_lvl1, hidden_channels[0], hidden_channels[1], num_layers, ResidualBlock, R)
        self.graphmaxpool1 = GraphPooling(
            graph_lvl1.nx3, graph_lvl2.nx3, graph_lvl1.nx2, graph_lvl2.nx2, graph_lvl1.nx1, graph_lvl2.nx1
        )
        self.block2 = NetworkBlock(graph_lvl2, hidden_channels[1], hidden_channels[2], num_layers, ResidualBlock, R)
        self.graphmaxpool2 = GraphPooling(
            graph_lvl2.nx3, graph_lvl3.nx3, graph_lvl2.nx2, graph_lvl3.nx2, graph_lvl2.nx1, graph_lvl3.nx1
        )
        self.block3 = NetworkBlock(graph_lvl3, hidden_channels[2], hidden_channels[3], num_layers, ResidualBlock, R)

        # output layer : global average pooling + fc
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_channels[3], out_channels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """

        B, _, _ = x.shape

        out = self.conv(x)

        out = self.block1(out)
        out = self.graphmaxpool1(out)
        out = self.block2(out)
        out = self.graphmaxpool2(out)
        out = self.block3(out)

        out = self.globalmaxpool(out).contiguous().view(B, -1)
        out = self.fc(out)
        out = self.logsoftmax(out)

        return out

    @property
    def capacity(self) -> int:
        """
        Returns the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): number of trainable parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())
