# coding=utf-8

import torch
from torch import nn

from ...graphs.graphs import Graph
from ..layers.blocks import NetworkBlock, ResidualBlock
from ..layers.convs import ChebConv


class WideResGEChebNet(nn.Module):
    """
    A Wide Residual Group Equivariant ChebNet for image classification.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        pool,
        graph_lvl0,
        graph_lvl1=None,
        graph_lvl2=None,
        depth=8,
        widen_factor=1,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): order of the Chebyshev polynomials.
            pool (`torch.nn.Module`): pooling layers.
            graph_lvl0 (`Graph`): graph at level 0, the coarsenest graph.
            graph_lvl1 (`Graph`): graph at level 1. Defaults to None.
            graph_lvl2 (`Graph`): graph at level 2, the finest graph. Defaults to None.
            depth (int): depth of the neural network. Defaults to 8.
            widen_factor (int, optional): widen factor of the neural network. Defaults to 1.

        Raises:
            ValueError: depth must be compatible with the architecture.
        """
        super(WideResGEChebNet, self).__init__()

        if (depth - 2) % 6:
            raise ValueError(f"{depth} is not a valid value for depth")

        if pool and (graph_lvl1 is None or graph_lvl2 is None):
            raise ValueError(f"Incompatible value for pool and graphs")

        hidden_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        num_layers = (depth - 2) // 6

        self.conv = ChebConv(
            in_channels,
            hidden_channels[0],
            kernel_size=kernel_size,
            bias=True,
            graph=graph_lvl2 if pool else graph_lvl0,
        )
        self.relu = nn.ReLU(inplace=True)

        self.pool2_1 = None if pool is None else pool(kernel_size=(1, 2), size=graph_lvl2.dim)
        self.pool1_0 = None if pool is None else pool(kernel_size=(1, 2), size=graph_lvl1.dim)

        self.block2 = NetworkBlock(
            hidden_channels[0],
            hidden_channels[1],
            num_layers,
            ResidualBlock,
            ChebConv,
            kernel_size,
            graph=graph_lvl2 if pool else graph_lvl0,
        )
        self.block1 = NetworkBlock(
            hidden_channels[1],
            hidden_channels[2],
            num_layers,
            ResidualBlock,
            ChebConv,
            kernel_size,
            graph=graph_lvl1 if pool else graph_lvl0,
        )
        self.block0 = NetworkBlock(
            hidden_channels[2], hidden_channels[3], num_layers, ResidualBlock, ChebConv, kernel_size, graph=graph_lvl0
        )

        # output layer : global average pooling + fc
        self.globalpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_channels[3], out_channels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): output tensor.
        """

        B, *_ = x.shape

        out = self.conv(x)

        out = self.block2(out)
        if not self.pool2_1 is None:
            out = self.pool2_1(out)
        out = self.block1(out)
        if not self.pool1_0 is None:
            out = self.pool1_0(out)
        out = self.block0(out)

        out = self.globalpool(out).contiguous().view(B, -1)
        out = self.fc(out)
        return self.logsoftmax(out)


class ChebEncoder(nn.Module):
    """
    A Chebyschev encoder consisting in sequential Chebyschev convolution plus pooling layers.
    """

    def __init__(
        self,
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
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): order of the Chebyshev polynomials.
            pool (`torch.nn.Module`): pooling layers.
            graph_lvl0 (`Graph`): graph at level 0, the coarsenest graph.
            graph_lvl1 (`Graph`): graph at level 1.
            graph_lvl2 (`Graph`): graph at level 2.
            graph_lvl3 (`Graph`): graph at level 3.
            graph_lvl4 (`Graph`): graph at level 4.
            graph_lvl5 (`Graph`): graph at level 5, the finest graph.
        """
        super(ChebEncoder, self).__init__()

        self.conv = ChebConv(in_channels, 16, kernel_size=kernel_size, bias=True, graph=graph_lvl5)
        self.relu = nn.ReLU(inplace=True)
        self.block5 = ResidualBlock(16, 32, ChebConv, kernel_size, graph=graph_lvl5)

        self.pool5_4 = pool(kernel_size=(1, 2), size=graph_lvl5.dim)
        self.block4 = ResidualBlock(32, 64, ChebConv, kernel_size, graph=graph_lvl4)

        self.pool4_3 = pool(kernel_size=(1, 2), size=graph_lvl4.dim)
        self.block3 = ResidualBlock(64, 128, ChebConv, kernel_size, graph=graph_lvl3)

        self.pool3_2 = pool(kernel_size=(1, 2), size=graph_lvl3.dim)
        self.block2 = ResidualBlock(128, 256, ChebConv, kernel_size, graph=graph_lvl2)

        self.pool2_1 = pool(kernel_size=(1, 2), size=graph_lvl2.dim)
        self.block1 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl1)

        self.pool1_0 = pool(kernel_size=(1, 2), size=graph_lvl1.dim)
        self.block0 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl0)

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): encoded tensor.
        """
        x_enc5 = self.block5(self.relu(self.conv(x)))
        x_enc4 = self.block4(self.pool5_4(x_enc5))
        x_enc3 = self.block3(self.pool4_3(x_enc4))
        x_enc2 = self.block2(self.pool3_2(x_enc3))
        x_enc1 = self.block1(self.pool2_1(x_enc2))
        x_enc0 = self.block0(self.pool1_0(x_enc1))

        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4, x_enc5


class ChebDecoder(nn.Module):
    """
    A Chebyschev decoder consisting in sequential Chebyschev convolution plus unpooling layers.
    """

    def __init__(
        self,
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
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): order of the Chebyshev polynomials.
            pool (`torch.nn.Module`): pooling layers.
            unpool (`torch.nn.Module`): unpooling layers.
            graph_lvl0 (`Graph`): graph at level 0, the coarsenest graph.
            graph_lvl1 (`Graph`): graph at level 1.
            graph_lvl2 (`Graph`): graph at level 2.
            graph_lvl3 (`Graph`): graph at level 3.
            graph_lvl4 (`Graph`): graph at level 4.
            graph_lvl5 (`Graph`): graph at level 5, the finest graph.
        """
        super(ChebDecoder, self).__init__()

        self.unpool0_1 = unpool(kernel_size=(1, 2), size=graph_lvl0.dim)
        self.unpool1_2 = unpool(kernel_size=(1, 2), size=graph_lvl1.dim)
        self.unpool2_3 = unpool(kernel_size=(1, 2), size=graph_lvl2.dim)
        self.unpool3_4 = unpool(kernel_size=(1, 2), size=graph_lvl3.dim)
        self.unpool4_5 = unpool(kernel_size=(1, 2), size=graph_lvl4.dim)

        self.block0 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl0)
        self.block1 = ResidualBlock(512, 256, ChebConv, kernel_size, graph=graph_lvl1)
        self.block2 = ResidualBlock(512, 128, ChebConv, kernel_size, graph=graph_lvl2)
        self.block3 = ResidualBlock(256, 64, ChebConv, kernel_size, graph=graph_lvl3)
        self.block4 = ResidualBlock(128, 32, ChebConv, kernel_size, graph=graph_lvl4)
        self.block5 = ResidualBlock(64, 16, ChebConv, kernel_size, graph=graph_lvl5)

        # pool on layers to break the symmetry axis
        self.pool5 = pool((graph_lvl5.dim[0], 1), graph_lvl5.dim)
        self.conv = ChebConv(16, out_channels, kernel_size=1, bias=False, graph=graph_lvl5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x_enc0, x_enc1, x_enc2, x_enc3, x_enc4, x_enc5):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): decoded tensor.
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
    A U-Net like ChebNet architecture for image segmentation.
    """

    def __init__(
        self,
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
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): order of the Chebyshev polynomials.
            pool (`torch.nn.Module`): pooling layers.
            unpool (`torch.nn.Module`): unpooling layers.
            graph_lvl0 (`Graph`): graph at level 0, the coarsenest graph.
            graph_lvl1 (`Graph`): graph at level 1.
            graph_lvl2 (`Graph`): graph at level 2.
            graph_lvl3 (`Graph`): graph at level 3.
            graph_lvl4 (`Graph`): graph at level 4.
            graph_lvl5 (`Graph`): graph at level 5, the finest graph.
        """
        super(UChebNet, self).__init__()

        self.encoder = ChebEncoder(
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
        self.decoder = ChebDecoder(
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

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): output tensor.
        """

        x_enc0, x_enc1, x_enc2, x_enc3, x_enc4, x_enc5 = self.encoder(x)
        x = self.decoder(x_enc0, x_enc1, x_enc2, x_enc3, x_enc4, x_enc5)
        return x
