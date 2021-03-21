# coding=utf-8

import torch
from torch import nn

from ...graphs.graphs import Graph
from ..layers.blocks import NetworkBlock, ResidualBlock
from ..layers.convs import ChebConv
from ..layers.pools import GlobalPool, SE2SpatialPool, SO3OrientationPool, SO3SpatialPool
from ..layers.unpools import SO3SpatialUnpool


class WideResSE2GEChebNet(nn.Module):
    """
    A Wide Residual Group Equivariant ChebNet for image classification.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        graph_lvl0,
        graph_lvl1=None,
        graph_lvl2=None,
        depth=8,
        widen_factor=1,
        reduction=None,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): order of the Chebyshev polynomials.
            reduction (str): pooling reduction operation in "max", "avg" or "rand".
            graph_lvl0 (`Graph`): graph at level 0, the coarsenest graph.
            graph_lvl1 (`Graph`): graph at level 1. Defaults to None.
            graph_lvl2 (`Graph`): graph at level 2, the finest graph. Defaults to None.
            depth (int): depth of the neural network. Defaults to 8.
            widen_factor (int, optional): widen factor of the neural network. Defaults to 1.

        Raises:
            ValueError: depth must be compatible with the architecture.
        """
        super(WideResSE2GEChebNet, self).__init__()

        if (depth - 2) % 6:
            raise ValueError(f"{depth} is not a valid value for depth")

        if not reduction is None and (graph_lvl1 is None or graph_lvl2 is None):
            raise ValueError(f"Incompatible value for pool and graphs")

        hidden_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        num_layers = (depth - 2) // 6

        self.conv = ChebConv(
            in_channels,
            hidden_channels[0],
            kernel_size=kernel_size,
            bias=True,
            graph=graph_lvl0 if reduction is None else graph_lvl2,
        )
        self.relu = nn.ReLU(inplace=True)

        self.pool2_1 = None if reduction is None else SE2SpatialPool(2, graph_lvl2.size, reduction)
        self.pool1_0 = None if reduction is None else SE2SpatialPool(2, graph_lvl1.size, reduction)

        self.block2 = NetworkBlock(
            hidden_channels[0],
            hidden_channels[1],
            num_layers,
            ResidualBlock,
            ChebConv,
            kernel_size,
            graph=graph_lvl0 if reduction is None else graph_lvl2,
        )
        self.block1 = NetworkBlock(
            hidden_channels[1],
            hidden_channels[2],
            num_layers,
            ResidualBlock,
            ChebConv,
            kernel_size,
            graph=graph_lvl0 if reduction is None else graph_lvl1,
        )
        self.block0 = NetworkBlock(
            hidden_channels[2], hidden_channels[3], num_layers, ResidualBlock, ChebConv, kernel_size, graph=graph_lvl0
        )

        # output layer : global max pooling + fc
        self.globalpool = GlobalPool(graph_lvl0.num_nodes, "max")
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


class SO3GEChebEncoder(nn.Module):
    """
    A Chebyschev encoder consisting in sequential Chebyschev convolution plus pooling layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        graph_lvl0,
        graph_lvl1,
        graph_lvl2,
        graph_lvl3,
        graph_lvl4,
        graph_lvl5,
        reduction,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): order of the Chebyshev polynomials.
            graph_lvl0 (`Graph`): graph at level 0, the coarsenest graph.
            graph_lvl1 (`Graph`): graph at level 1.
            graph_lvl2 (`Graph`): graph at level 2.
            graph_lvl3 (`Graph`): graph at level 3.
            graph_lvl4 (`Graph`): graph at level 4.
            graph_lvl5 (`Graph`): graph at level 5, the finest graph.
        """
        super(SO3GEChebEncoder, self).__init__()

        self.conv = ChebConv(in_channels, 16, kernel_size=kernel_size, bias=True, graph=graph_lvl5)
        self.relu = nn.ReLU(inplace=True)
        self.block5 = ResidualBlock(16, 32, ChebConv, kernel_size, graph=graph_lvl5)

        self.pool5_4 = SO3SpatialPool(2, graph_lvl5.size, reduction)
        self.block4 = ResidualBlock(32, 64, ChebConv, kernel_size, graph=graph_lvl4)

        self.pool4_3 = SO3SpatialPool(2, graph_lvl4.size, reduction)
        self.block3 = ResidualBlock(64, 128, ChebConv, kernel_size, graph=graph_lvl3)

        self.pool3_2 = SO3SpatialPool(2, graph_lvl3.size, reduction)
        self.block2 = ResidualBlock(128, 256, ChebConv, kernel_size, graph=graph_lvl2)

        self.pool2_1 = SO3SpatialPool(2, graph_lvl2.size, reduction)
        self.block1 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl1)

        self.pool1_0 = SO3SpatialPool(2, graph_lvl1.size, reduction)
        self.block0 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl0)

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): encoded tensor.
        """
        print(x.shape)
        print(self.conv.graph.size)
        x_enc5 = self.block5(self.relu(self.conv(x)))
        x_enc4 = self.block4(self.pool5_4(x_enc5))
        x_enc3 = self.block3(self.pool4_3(x_enc4))
        x_enc2 = self.block2(self.pool3_2(x_enc3))
        x_enc1 = self.block1(self.pool2_1(x_enc2))
        x_enc0 = self.block0(self.pool1_0(x_enc1))

        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4, x_enc5


class SO3GEChebDecoder(nn.Module):
    """
    A Chebyschev decoder consisting in sequential Chebyschev convolution plus unpooling layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        graph_lvl0,
        graph_lvl1,
        graph_lvl2,
        graph_lvl3,
        graph_lvl4,
        graph_lvl5,
        reduction,
        expansion,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): order of the Chebyshev polynomials.
            graph_lvl0 (`Graph`): graph at level 0, the coarsenest graph.
            graph_lvl1 (`Graph`): graph at level 1.
            graph_lvl2 (`Graph`): graph at level 2.
            graph_lvl3 (`Graph`): graph at level 3.
            graph_lvl4 (`Graph`): graph at level 4.
            graph_lvl5 (`Graph`): graph at level 5, the finest graph.
        """
        super(SO3GEChebDecoder, self).__init__()

        self.unpool0_1 = SO3SpatialUnpool(2, graph_lvl0.size, expansion)
        self.unpool1_2 = SO3SpatialUnpool(2, graph_lvl1.size, expansion)
        self.unpool2_3 = SO3SpatialUnpool(2, graph_lvl2.size, expansion)
        self.unpool3_4 = SO3SpatialUnpool(2, graph_lvl3.size, expansion)
        self.unpool4_5 = SO3SpatialUnpool(2, graph_lvl4.size, expansion)

        self.block0 = ResidualBlock(256, 256, ChebConv, kernel_size, graph=graph_lvl0)
        self.block1 = ResidualBlock(512, 256, ChebConv, kernel_size, graph=graph_lvl1)
        self.block2 = ResidualBlock(512, 128, ChebConv, kernel_size, graph=graph_lvl2)
        self.block3 = ResidualBlock(256, 64, ChebConv, kernel_size, graph=graph_lvl3)
        self.block4 = ResidualBlock(128, 32, ChebConv, kernel_size, graph=graph_lvl4)
        self.block5 = ResidualBlock(64, 16, ChebConv, kernel_size, graph=graph_lvl5)

        # pool on layers to break the symmetry axis
        self.pool5 = SO3OrientationPool(graph_lvl5.size[-1], graph_lvl5.size, reduction)
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


class SO3GEUChebNet(nn.Module):
    """
    A U-Net like spherical ChebNet architecture for image segmentation on the sphere.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        graph_lvl0,
        graph_lvl1,
        graph_lvl2,
        graph_lvl3,
        graph_lvl4,
        graph_lvl5,
        reduction,
        expansion,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): order of the Chebyshev polynomials.
            graph_lvl0 (`Graph`): graph at level 0, the coarsenest graph.
            graph_lvl1 (`Graph`): graph at level 1.
            graph_lvl2 (`Graph`): graph at level 2.
            graph_lvl3 (`Graph`): graph at level 3.
            graph_lvl4 (`Graph`): graph at level 4.
            graph_lvl5 (`Graph`): graph at level 5, the finest graph.
        """
        super(SO3GEUChebNet, self).__init__()

        self.encoder = SO3GEChebEncoder(
            in_channels,
            out_channels,
            kernel_size,
            graph_lvl0,
            graph_lvl1,
            graph_lvl2,
            graph_lvl3,
            graph_lvl4,
            graph_lvl5,
            reduction,
        )
        self.decoder = SO3GEChebDecoder(
            in_channels,
            out_channels,
            kernel_size,
            graph_lvl0,
            graph_lvl1,
            graph_lvl2,
            graph_lvl3,
            graph_lvl4,
            graph_lvl5,
            reduction,
            expansion,
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
