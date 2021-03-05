# coding=utf-8

import torch
from torch import nn


class NetworkBlock(nn.Module):
    """
    A neural network block consisting in a sequence of convolutional layer blocks.
    """

    def __init__(self, in_channels, out_channels, num_layers, block, conv, kernel_size, *args, **kwargs):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            num_layers (int): number of layers of the network block.
            block (`torch.nn.Module`): type of block constituting the network block.
            conv (`torch.nn.Module`): convolutional layer.
            kernel_size (int): kernel size.
        """
        super(NetworkBlock, self).__init__()
        self.layers = nn.Sequential(
            *[
                block(out_channels if i > 0 else in_channels, out_channels, conv, kernel_size, *args, **kwargs)
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): output tensor.
        """
        return self.layers(x)


class BasicBlock(nn.Module):
    """
    A basic neural network block with batch normalization, convolutional layer and ReLU activation function.
    """

    def __init__(self, in_channels, out_channels, conv, kernel_size, *args, **kwargs):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            conv (`torch.nn.Module`): convolutional layer.
            kernel_size (int): kernel size.
        """
        super(BasicBlock, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size, bias=True, *args, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): output tensor.
        """
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    A residual neural network block with batch normalization, convolutional layers and ReLU activation function.
    """

    def __init__(self, in_channels, out_channels, conv, kernel_size, *args, **kwargs):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            conv (`torch.nn.Module`): convolutional layer.
            kernel_size (int): kernel size.
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = conv(in_channels, out_channels, kernel_size, bias=True, *args, **kwargs)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(out_channels, out_channels, kernel_size, bias=True, *args, **kwargs)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.equalInOut = in_channels == out_channels

        if not self.equalInOut:
            self.convShortcut = conv(in_channels, out_channels, kernel_size=1, bias=False, *args, **kwargs)

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): input tensor.

        Returns:
            (`torch.Tensor`): output tensor.
        """
        out = self.relu1(self.bn1(self.conv1(x)))

        if self.equalInOut:
            return self.relu2(self.bn2(x + self.conv2(out)))

        return self.relu2(self.bn2(self.convShortcut(x) + self.conv2(out)))
