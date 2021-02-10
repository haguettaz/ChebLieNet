import torch
from torch.nn import BatchNorm1d, Module, ModuleList, ReLU

from .convolution import ChebConv


class NetworkBlock(Module):
    def __init__(self, in_channels, out_channels, num_layers, block, K):
        super(NetworkBlock, self).__init__()
        self.layers = ModuleList(
            [block(out_channels if i > 0 else in_channels, out_channels, K) for i in range(num_layers)]
        )

    def forward(self, x, laplacian):
        for l in self.layers:
            x = l(x, laplacian)
        return x


class BasicBlock(Module):
    def __init__(self, in_channels, out_channels, K):
        super(BasicBlock, self).__init__()
        self.bn = BatchNorm1d(in_channels)
        self.conv = ChebConv(in_channels, out_channels, K, bias=True)
        self.relu = ReLU(inplace=True)

    def forward(self, x, laplacian):
        return self.relu(self.conv(self.bn(x), laplacian))


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, K):
        super(ResidualBlock, self).__init__()

        self.bn1 = BatchNorm1d(in_channels)
        self.relu1 = ReLU(inplace=True)
        self.conv1 = ChebConv(in_channels, out_channels, K, bias=True)
        self.bn2 = BatchNorm1d(out_channels)
        self.relu2 = ReLU(inplace=True)
        self.conv2 = ChebConv(out_channels, out_channels, K, bias=True)

        self.equalInOut = in_channels == out_channels

        if not self.equalInOut:
            self.convShortcut = ChebConv(in_channels, out_channels, K=1, bias=False)

    def forward(self, x, laplacian):
        x = self.bn1(x)
        out = self.relu1(self.conv1(x, laplacian))
        if self.equalInOut:
            return self.relu2(x + self.conv2(self.bn2(out), laplacian))
        return self.relu2(self.convShortcut(x) + self.conv2(self.bn2(out), laplacian))
