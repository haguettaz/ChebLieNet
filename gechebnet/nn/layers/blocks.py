import torch
from torch import Tensor, nn

from ...graphs.graphs import Graph
from .convs import ChebConv


class NetworkBlock(nn.Module):
    """
    Network block with 2d convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        block: nn.Module,
        conv: nn.Module,
        kernel_size: int,
        *args,
        **kwargs
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
        super(NetworkBlock, self).__init__()
        self.layers = nn.Sequential(
            *[
                block(out_channels if i > 0 else in_channels, out_channels, conv, kernel_size * args, **kwargs)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """
        return self.layers(x)


class BasicBlock(nn.Module):
    """
    Basic block composed of batch normalization, convolutional layer and ReLU activation function.
    """

    def __init__(self, in_channels: int, out_channels: int, conv: nn.Module, kernel_size: int, *args, **kwargs):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            conv (nn.Module): convolutional layer.
            kernel_size (int): kernel size.
        """
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = conv(in_channels, out_channels, kernel_size, bias=True, *args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """
        return self.relu(self.conv(self.bn(x)))


class ResidualBlock(nn.Module):
    """
    Residual block composed of batch normalization, 2 convolutional layers and ReLU activation function.
    """

    def __init__(self, in_channels: int, out_channels: int, conv: nn.Module, kernel_size: int, *args, **kwargs):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            conv (nn.Module): convolutional layer.
            kernel_size (int): kernel size.
        """
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = conv(in_channels, out_channels, kernel_size, bias=True, *args, **kwargs)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = conv(out_channels, out_channels, kernel_size, bias=True, *args, **kwargs)

        self.equalInOut = in_channels == out_channels

        if not self.equalInOut:
            self.convShortcut = conv(in_channels, out_channels, kernel_size=1, bias=False, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """
        x = self.bn1(x)
        out = self.relu1(self.conv1(x))
        
        if self.equalInOut:
            return self.relu2(x + self.conv2(self.bn2(out)))

        return self.relu2(self.convShortcut(x) + self.conv2(self.bn2(out)))
