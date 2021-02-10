import torch
from torch import nn

from .convolution import ChebConv


class NetworkBlock(nn.Module):
    def __init__(self, graph, in_channels, out_channels, num_layers, block, K):
        super(NetworkBlock, self).__init__()
        self.layers = nn.Sequential(
            *[block(graph, out_channels if i > 0 else in_channels, out_channels, K) for i in range(num_layers)]
        )

    def forward(self, x):
        return self.layers(x)


class BasicBlock(nn.Module):
    def __init__(self, graph, in_channels, out_channels, K):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = ChebConv(graph, in_channels, out_channels, K, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(self.bn(x)))


class ResidualBlock(nn.Module):
    def __init__(self, graph, in_channels, out_channels, K):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = ChebConv(graph, in_channels, out_channels, K, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = ChebConv(graph, out_channels, out_channels, K, bias=True)

        self.equalInOut = in_channels == out_channels

        if not self.equalInOzut:
            self.convShortcut = ChebConv(graph, in_channels, out_channels, K=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        out = self.relu1(self.conv1(x))
        if self.equalInOut:
            return self.relu2(x + self.conv2(self.bn2(out)))
        return self.relu2(self.convShortcut(x) + self.conv2(self.bn2(out)))
