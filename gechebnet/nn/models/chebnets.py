from typing import Optional

import torch
from torch import Tensor, nn

from ...graphs.graphs import Graph
from ..layers.blocks import NetworkBlock, ResidualBlock
from ..layers.convs import ChebConv


class WideResGEChebNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        R: int,
        pool: nn.Module,
        graph_lvl0: Graph,
        graph_lvl1: Graph,
        graph_lvl2: Graph,
        depth: int,
        widen_factor: Optional[int] = 1,
    ):
        """
        Initialization.

        Args:
            graph_lvl0 (Graph): graph at level 0.
            graph_lvl1 (Graph): graph at level 1.
            graph_lvl2 (Graph): graph at level 2.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            R (int): order of the Chebyshev polynomials.
            depth (int): depth of the neural network.
            widen_factor (int, optional): widen factor of the neural network. Defaults to 1.

        Raises:
            ValueError: depth must be compatible with the architecture.
        """
        super(WideResGEChebNet, self).__init__()

        if (depth - 2) % 6:
            raise ValueError(f"{depth} is not a valid value for {depth}")

        hidden_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        num_layers = (depth - 2) // 6

        self.conv = ChebConv(in_channels, hidden_channels[0], kernel_size=R, bias=True, graph=graph_lvl0)
        self.relu = nn.ReLU(inplace=True)

        self.pool0_1 = None if pool is None else pool(kernel_size=(1, 2), size=graph_lvl0.size)
        self.pool1_2 = None if pool is None else pool(kernel_size=(1, 2), size=graph_lvl1.size)

        self.block2 = NetworkBlock(
            hidden_channels[0], hidden_channels[1], num_layers, ResidualBlock, ChebConv, R, graph=graph_lvl0
        )
        self.block1 = NetworkBlock(
            hidden_channels[1], hidden_channels[2], num_layers, ResidualBlock, ChebConv, R, graph=graph_lvl1
        )
        self.block0 = NetworkBlock(
            hidden_channels[2], hidden_channels[3], num_layers, ResidualBlock, ChebConv, R, graph=graph_lvl2
        )

        # output layer : global average pooling + fc
        self.globalpool = nn.AdaptiveMaxPool1d(1)
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

        B, *_ = x.shape

        out = self.conv(x)

        out = self.block2(out)
        out = out if self.pool2_1 is None else self.pool2_1(out)
        out = self.block1(out)
        out = out if self.pool1_0 is None else self.pool1_0(out)
        out = self.block0(out)

        out = self.globalpool(out).contiguous().view(B, -1)
        out = self.fc(out)
        return self.logsoftmax(x)

    @property
    def capacity(self) -> int:
        """
        Returns the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): number of trainable parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())


class Encoder(nn.Module):
    """
    Basic class for encoder network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool: nn.Module,
        graph_lvl0: Graph,
        graph_lvl1: Graph,
        graph_lvl2: Graph,
        graph_lvl3: Graph,
        graph_lvl4: Graph,
        graph_lvl5: Graph,
    ):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            num_layers (int): number of layers of the network block.
            block (nn.Module): type of block constituting the network block.
            conv (nn.Module): convolutional layer.
            kernel_size (int): kernel size.
        """
        super(Encoder, self).__init__()

        self.conv = ChebConv(in_channels, 16, kernel_size=kernel_size, bias=True, graph=graph_lvl5)
        self.relu = nn.ReLU(inplace=True)
        self.block5 = ResidualBlock(16, 32, ChebConv, kernel_size, graph=graph_lvl5)

        self.pool5_4 = pool(kernel_size=(1, 2), size=graph_lvl5.size)
        self.block4 = ResidualBlock(32, 64, ChebConv, kernel_size, graph=graph_lvl4)

        self.pool4_3 = pool(kernel_size=(1, 2), size=graph_lvl4.size)
        self.block3 = ResidualBlock(64, 128, ChebConv, kernel_size, graph=graph_lvl3)

        self.pool3_2 = pool(kernel_size=(1, 2), size=graph_lvl3.size)
        self.block2 = ResidualBlock(128, 256, ChebConv, kernel_size, graph=graph_lvl2)

        self.pool2_1 = pool(kernel_size=(1, 2), size=graph_lvl2.size)
        self.block1 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl1)

        self.pool1_0 = pool(kernel_size=(1, 2), size=graph_lvl1.size)
        self.block0 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """
        x_enc5 = self.block5(self.relu(self.conv(x)))
        x_enc4 = self.block4(self.pool5_4(x_enc5))
        x_enc3 = self.block3(self.pool4_3(x_enc4))
        x_enc2 = self.block2(self.pool3_2(x_enc3))
        x_enc1 = self.block1(self.pool2_1(x_enc2))
        x_enc0 = self.block0(self.pool1_0(x_enc1))

        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4, x_enc5


class Decoder(nn.Module):
    """
    Basic class for encoder network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool: nn.Module,
        unpool: nn.Module,
        graph_lvl0: Graph,
        graph_lvl1: Graph,
        graph_lvl2: Graph,
        graph_lvl3: Graph,
        graph_lvl4: Graph,
        graph_lvl5: Graph,
    ):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            num_layers (int): number of layers of the network block.
            block (nn.Module): type of block constituting the network block.
            conv (nn.Module): convolutional layer.
            kernel_size (int): kernel size.
        """
        super(Decoder, self).__init__()

        self.unpool0_1 = unpool(kernel_size=(1, 2), size=graph_lvl0.size)
        self.unpool1_2 = unpool(kernel_size=(1, 2), size=graph_lvl1.size)
        self.unpool2_3 = unpool(kernel_size=(1, 2), size=graph_lvl2.size)
        self.unpool3_4 = unpool(kernel_size=(1, 2), size=graph_lvl3.size)
        self.unpool4_5 = unpool(kernel_size=(1, 2), size=graph_lvl4.size)

        self.block0 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl0)
        self.block1 = ResidualBlock(512, 256, ChebConv, kernel_size, graph=graph_lvl1)
        self.block2 = ResidualBlock(512, 128, ChebConv, kernel_size, graph=graph_lvl2)
        self.block3 = ResidualBlock(256, 64, ChebConv, kernel_size, graph=graph_lvl3)
        self.block4 = ResidualBlock(128, 32, ChebConv, kernel_size, graph=graph_lvl4)
        self.block5 = ResidualBlock(64, 16, ChebConv, kernel_size, graph=graph_lvl5)

        # pool the symmetry axis
        self.pool5 = pool((graph_lvl5.size[0], 1), graph_lvl5.size)
        self.conv = ChebConv(16, out_channels, kernel_size=1, bias=False, graph=graph_lvl5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x_enc0: Tensor, x_enc1: Tensor, x_enc2: Tensor, x_enc3: Tensor, x_enc4: Tensor, x_enc5) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """
        x_dec1 = self.block1(torch.cat((self.unpool0_1(x_enc0), x_enc1), dim=1))
        x_dec2 = self.block2(torch.cat((self.unpool1_2(x_dec1), x_enc2), dim=1))
        x_dec3 = self.block3(torch.cat((self.unpool2_3(x_dec2), x_enc3), dim=1))
        x_dec4 = self.block4(torch.cat((self.unpool3_4(x_dec3), x_enc4), dim=1))
        x_dec5 = self.block5(torch.cat((self.unpool4_5(x_dec4), x_enc5), dim=1))
        x = self.conv(self.pool5(x_dec5))

        return self.logsoftmax(x)


class UChebNet(nn.Module):
    """
    Basic class for encoder network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool: nn.Module,
        unpool: nn.Module,
        graph_lvl0: Graph,
        graph_lvl1: Graph,
        graph_lvl2: Graph,
        graph_lvl3: Graph,
        graph_lvl4: Graph,
        graph_lvl5: Graph,
    ):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            num_layers (int): number of layers of the network block.
            block (nn.Module): type of block constituting the network block.
            conv (nn.Module): convolutional layer.
            kernel_size (int): kernel size.
        """
        super(UChebNet, self).__init__()

        self.encoder = Encoder(
            in_channels,
            out_channels,
            kernel_size,
            pool,
            graph_lvl0,
            graph_lvl1,
            graph_lvl2,
            graph_lvl3,
            graph_lvl4,
            graph_lvl5,
        )
        self.decoder = Decoder(
            in_channels,
            out_channels,
            kernel_size,
            pool,
            unpool,
            graph_lvl0,
            graph_lvl1,
            graph_lvl2,
            graph_lvl3,
            graph_lvl4,
            graph_lvl5,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """

        x_enc0, x_enc1, x_enc2, x_enc3, x_enc4, x_enc5 = self.encoder(x)
        x = self.decoder(x_enc0, x_enc1, x_enc2, x_enc3, x_enc4, x_enc5)
        return x
