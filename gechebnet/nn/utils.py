import torch
from torch import Tensor, nn

from ..graph.graph import Graph
from .convolution import ChebConv


class ChebNetworkBlock(nn.Module):
    """
    Network block.
    """

    def __init__(self, graph: Graph, in_channels: int, out_channels: int, num_layers: int, block: nn.Module, R: int):
        """
        Initialization.

        Args:
            graph (Graph): graph.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            num_layers (int): number of layers of the network block.
            block (nn.Module): type of block to constitute the network block.
            R (int): order of the Chebyshev polynomials.
        """
        super(ChebNetworkBlock, self).__init__()
        self.layers = nn.Sequential(
            *[block(graph, out_channels if i > 0 else in_channels, out_channels, R) for i in range(num_layers)]
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


class ChebBasicBlock(nn.Module):
    """
    Basic block composed of batch normalization, convolutional layer and ReLU activation function.
    """

    def __init__(self, graph: Graph, in_channels: int, out_channels: int, R: int):
        """
        Initialization.

        Args:
            graph (Graph): graph.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            R (int): order of the Chebyshev polynomials.
        """
        super(ChebBasicBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = ChebConv(graph, in_channels, out_channels, R, bias=True)
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


class ChebResidualBlock(nn.Module):
    """
    Residual block composed of batch normalization, 2 convolutional layers and ReLU activation function.
    """

    def __init__(self, graph: Graph, in_channels: int, out_channels: int, R: int):
        """
        Initialization.

        Args:
            graph (Graph): graph.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            R (int): order of the Chebyshev polynomials.
        """
        super(ChebResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = ChebConv(graph, in_channels, out_channels, R, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = ChebConv(graph, out_channels, out_channels, R, bias=True)

        self.equalInOut = in_channels == out_channels

        if not self.equalInOut:
            self.convShortcut = ChebConv(graph, in_channels, out_channels, R=1, bias=False)

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


class NetworkBlock(nn.Module):
    """
    Network block.
    """

    def __init__(self, in_channels: int, out_channels: int, num_layers: int, block: nn.Module, kernel_size: int):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            num_layers (int): number of layers of the network block.
            block (nn.Module): type of block to constitute the network block.
            kernel_size (int): kernel size.
        """
        super(NetworkBlock, self).__init__()
        self.layers = nn.Sequential(
            *[block(out_channels if i > 0 else in_channels, out_channels, kernel_size) for i in range(num_layers)]
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

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kerne_size (int): kernel size.
        """
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
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

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        Initialization.

        Args:
            graph (Graph): graph.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
        """
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, bias=True)

        self.equalInOut = in_channels == out_channels

        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

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


class GraphPooling(nn.Module):
    """
    Graph pooling to project a signal from one fine graph to a coarsened graph.
    """

    def __init__(self, Lin: int, Lout: int, Hin: int, Hout: int, Win: int, Wout: int):
        """
        Initialization.

        Args:
            Lin (int): input depth.
            Lout (int): ouput depth.
            Hin (int): input height.
            Hout (int): output height.
            Win (int): input width.
            Wout (int): output width.
        """
        super(GraphPooling, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool3d((Lout, Hout, Wout))
        self.Lin, self.Hin, self.Win = Lin, Hin, Win

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input tensor.

        Returns:
            (Tensor): output tensor.
        """
        B, C, _ = x.shape
        return self.maxpool(x.view(B, C, self.Lin, self.Hin, self.Win)).view(B, C, -1)
