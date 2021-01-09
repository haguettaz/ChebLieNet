from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn import BatchNorm1d, Module

from ..graph.graph import Graph
from .convolution import ChebConv


class GEChebNet(Module):
    def __init__(
        self,
        graphs: Tuple[Graph, ...],
        K: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        laplacian_device: Optional[torch.device] = None,
        pooling: Optional[str] = "max",
    ):
        """
        Initialize a ChebNet with 6 convolutional layers and batch normalization.

        Args:
            graphs (tuple): tuple of graphs object, one for each pooling results.
            K (int): the degree of the Chebyschev polynomials, the sum goes from indices 0 to K-1.
            in_channels (int): the number of dimensions of the input layer.
            out_channels (int): the number of dimensions of the output layer.
            hidden_channels (int): the number of dimensions of the hidden layers.
            laplacian_device (torch.device, optional): computation device.
            pooling (str, optional): pooling type.

        Raises:
            ValueError: pooling must be in {'max', 'avg'}.
        """
        super(GEChebNet, self).__init__()

        if pooling not in {"max", "avg"}:
            raise ValueError(f"{pooling} is not a valid value for pooling: must be 'max' or 'avg'")

        laplacian_device = laplacian_device or torch.device("cpu")
        graph_1, graph_2, graph_3 = graphs

        self.conv1 = ChebConv(graph_1, in_channels, hidden_channels, K, laplacian_device=laplacian_device)
        self.conv2 = ChebConv(graph_1, hidden_channels, hidden_channels, K, laplacian_device=laplacian_device)

        self.conv3 = ChebConv(graph_2, hidden_channels, hidden_channels, K, laplacian_device=laplacian_device)
        self.conv4 = ChebConv(graph_2, hidden_channels, hidden_channels, K, laplacian_device=laplacian_device)

        self.conv5 = ChebConv(graph_3, hidden_channels, hidden_channels, K, laplacian_device=laplacian_device)
        self.conv6 = ChebConv(graph_3, hidden_channels, out_channels, K, laplacian_device=laplacian_device)

        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)
        self.bn6 = BatchNorm1d(hidden_channels)

        self.nx1 = [graph_1.nx1, graph_2.nx1, graph_3.nx1]
        self.nx2 = [graph_1.nx2, graph_2.nx2, graph_3.nx2]
        self.nx3 = [graph_1.nx3, graph_2.nx3, graph_3.nx3]

        if pooling == "max":
            self.pooling = F.max_pool3d
        else:
            self.pooling = F.avg_pool3d

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Forward function receiving as input a batch and outputing a prediction on this batch

        Args:
            x (FloatTensor): the batch to feed the network with.

        Returns:
            (FloatTensor): the predictions on the batch.
        """

        B, _, _ = x.shape

        # Chebyschev Convolutions
        x = self.conv1(x)  # (B, C, V)
        x = F.relu(x)
        x = self.bn2(x)  # (B, C, V)
        x = self.conv2(x)  # (B, C, V)
        x = F.relu(x)

        # Spatial pooling
        x = x.view(B, -1, self.nx3[0], self.nx2[0], self.nx1[0])  # (B, C, L, H, W)
        x = self.pooling(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, C, L, H', W')
        x = x.view(B, -1, self.nx3[1] * self.nx2[1] * self.nx1[1])  # (B, C, V)

        # Chebyschev convolutions
        x = self.bn3(x)  # (B, C, V)
        x = self.conv3(x)  # (B, C, V)
        x = F.relu(x)
        x = self.bn4(x)  # (B, C, V)
        x = self.conv4(x)  # (B, C, V)
        x = F.relu(x)

        # Spatial pooling
        x = x.view(B, -1, self.nx3[1], self.nx2[1], self.nx1[1])  # (B, C, L, H, W)
        x = self.pooling(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, C, L, H', W')
        x = x.view(B, -1, self.nx3[2] * self.nx2[2] * self.nx1[2])  # (B, C, V)

        # 2 convolutions
        x = self.bn5(x)  # (B, C, V)
        x = self.conv5(x)  # (B, C, V)
        x = F.relu(x)
        x = self.bn6(x)  # (B, C, V)
        x = self.conv6(x)  # (B, C, V)
        x = F.relu(x)

        # Global pooling
        x = x.view(B, -1, self.nx3[2], self.nx2[2], self.nx1[2])  # (B, C, L, H, W)
        x = self.pooling(x, kernel_size=(self.nx3[2], self.nx2[2], self.nx1[2]))  # (B, C, 1, 1, 1)
        x = x.view(B, -1)  # (B, C)

        x = F.log_softmax(x, dim=1)  # (B, C)

        return x

    @property
    def capacity(self) -> int:
        """
        Return the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): number of trainable parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())
