import torch
from torch import nn
from torch.nn import AdaptiveMaxPool3d

from .convolution import ChebConv


class NetworkBlock(nn.Module):
    def __init__(self, graph, in_channels, out_channels, num_layers, block, R):
        super(NetworkBlock, self).__init__()
        self.layers = nn.Sequential(
            *[block(graph, out_channels if i > 0 else in_channels, out_channels, R) for i in range(num_layers)]
        )

    def forward(self, x):
        return self.layers(x)


class BasicBlock(nn.Module):
    def __init__(self, graph, in_channels, out_channels, R):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = ChebConv(graph, in_channels, out_channels, R, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(self.bn(x)))


class ResidualBlock(nn.Module):
    def __init__(self, graph, in_channels, out_channels, R):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = ChebConv(graph, in_channels, out_channels, R, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = ChebConv(graph, out_channels, out_channels, R, bias=True)

        self.equalInOut = in_channels == out_channels

        if not self.equalInOut:
            self.convShortcut = ChebConv(graph, in_channels, out_channels, R=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        out = self.relu1(self.conv1(x))
        if self.equalInOut:
            return self.relu2(x + self.conv2(self.bn2(out)))
        return self.relu2(self.convShortcut(x) + self.conv2(self.bn2(out)))


class GraphPooling(nn.Module):
    def __init__(self, Lin, Lout, Hin, Hout, Win, Wout):
        super(GraphPooling, self).__init__()

        self.maxpool = AdaptiveMaxPool3d((Lout, Hout, Wout))
        self.Lin, self.Hin, self.Win = Lin, Hin, Win

    def forward(self, x):
        B, C, _ = x.shape
        return self.maxpool(x.view(B, C, self.Lin, self.Hin, self.Win)).view(B, C, -1)
