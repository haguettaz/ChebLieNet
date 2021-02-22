from typing import Optional

from torch import Tensor, nn

from ..graph.graph import Graph
from .convolution import ChebConv
from .utils import BasicBlock, NetworkBlock, ResidualBlock


class WideCNN(nn.Module):
    """
    Wide Group Equivariant ChebNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        depth: int,
        widen_factor: Optional[int] = 1,
        pooling: Optional[bool] = False,
    ):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            depth (int): depth of the neural network.
            widen_factor (int, optional): widen factor of the neural network. Defaults to 1.
            pooling (bool, optional): if True, add pooling layers at the end of each block. Defaults to False.

        Raises:
            ValueError: depth must be compatible with the architecture.
        """
        super(WideCNN, self).__init__()

        hidden_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        if (depth - 2) % 3:
            raise ValueError(f"{depth} is not a valid value for depth")

        num_layers = (depth - 2) // 3

        if pooling:
            self.maxpool1 = nn.MaxPool2d(2)
            self.maxpool2 = nn.MaxPool2d(2)
        else:
            self.maxpool1 = None
            self.maxpool2 = None

        # input layer : convolutional layer + relu
        self.conv = nn.Conv2d(in_channels, hidden_channels[0], kernel_size)
        self.relu = nn.ReLU(inplace=True)

        # hidden layers : 3 convolutional blocks + optional spatial pooling
        self.block1 = NetworkBlock(hidden_channels[0], hidden_channels[1], num_layers, BasicBlock, kernel_size)

        self.block2 = NetworkBlock(hidden_channels[1], hidden_channels[2], num_layers, BasicBlock, kernel_size)

        self.block3 = NetworkBlock(hidden_channels[2], hidden_channels[3], num_layers, BasicBlock, kernel_size)

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

        out = self.relu(self.conv(x))

        out = self.block1(out)
        if self.maxpool1 is not None:
            out = self.maxpool1(out)
        out = self.block2(out)
        if self.maxpool2 is not None:
            out = self.maxpool2(out)
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


class WideResCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        depth: int,
        widen_factor: Optional[int] = 1,
        pooling: Optional[bool] = False,
    ):
        """
        Initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            depth (int): depth of the neural network.
            widen_factor (int, optional): widen factor of the neural network. Defaults to 1.
            pooling (bool, optional): if True, add pooling layers at the end of each block. Defaults to False.

        Raises:
            ValueError: depth must be compatible with the architecture.
        """

        super(WideResCNN, self).__init__()

        hidden_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        if (depth - 2) % 3:
            raise ValueError(f"{depth} is not a valid value for depth")

        num_layers = (depth - 2) // 3

        if pooling:
            self.maxpool1 = nn.MaxPool2d(2)
            self.maxpool2 = nn.MaxPool2d(2)
        else:
            self.maxpool1 = None
            self.maxpool2 = None

        # input layer : convolutional layer + relu
        self.conv = nn.Conv2d(in_channels, hidden_channels[0], kernel_size)
        self.relu = nn.ReLU(inplace=True)

        # hidden layers : 3 convolutional blocks + optional spatial pooling
        self.block1 = NetworkBlock(hidden_channels[0], hidden_channels[1], num_layers, ResidualBlock, kernel_size)

        self.block2 = NetworkBlock(hidden_channels[1], hidden_channels[2], num_layers, ResidualBlock, kernel_size)

        self.block3 = NetworkBlock(hidden_channels[2], hidden_channels[3], num_layers, ResidualBlock, kernel_size)

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

        if self.maxpool1 is not None:
            out = self.maxpool1(out)
        out = self.block2(out)
        if self.maxpool2 is not None:
            out = self.maxpool2(out)
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
