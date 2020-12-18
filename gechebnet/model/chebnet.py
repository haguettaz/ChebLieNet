import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d

from .convolution import ChebConv


class ChebNet(torch.nn.Module):
    def __init__(self, graphs, K, in_channels, out_channels, hidden_channels, laplacian_device=None, pooling="max"):
        """
        Initialize a ChebNet with 6 convolutional layers and batch normalization.

        Args:
            K (int): the degree of the Chebyschev polynomials, the sum goes from indices 0 to K-1.
            num_layers (int): the number of layers on the orientation axis.
            input_dim (int, optional): the number of dimensions of the input layer. Defaults to 1.
            output_dim (int, optional): the number of dimensions of the output layer. Defaults to 10.
            hidden_dim (int, optional): the number of dimensions of the hidden layers. Defaults to 10.
        """
        super(ChebNet, self).__init__()

        laplacian_device = laplacian_device or torch.device("cpu")

        if pooling not in {"max", "avg"}:
            raise ValueError(f"{pooling} is not a valid value for pooling: must be 'max' or 'avg'")

        self.conv1 = ChebConv(graphs[0], in_channels, hidden_channels, K, laplacian_device=laplacian_device)
        self.conv2 = ChebConv(graphs[0], hidden_channels, hidden_channels, K, laplacian_device=laplacian_device)

        self.conv3 = ChebConv(graphs[1], hidden_channels, hidden_channels, K, laplacian_device=laplacian_device)
        self.conv4 = ChebConv(graphs[1], hidden_channels, hidden_channels, K, laplacian_device=laplacian_device)

        self.conv5 = ChebConv(graphs[2], hidden_channels, hidden_channels, K, laplacian_device=laplacian_device)
        self.conv6 = ChebConv(graphs[2], hidden_channels, out_channels, K, laplacian_device=laplacian_device)

        self.bn1 = BatchNorm1d(in_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)
        self.bn6 = BatchNorm1d(hidden_channels)

        self.nx1 = [graph.nx1 for graph in graphs]
        self.nx2 = [graph.nx2 for graph in graphs]
        self.nx3 = [graph.nx3 for graph in graphs]

        if pooling == "max":
            self.pooling = F.max_pool3d
        else:
            self.pooling = F.avg_pool3d

    def forward(self, x):
        """
        Forward function receiving as input a batch and outputing a prediction on this batch

        Args:
            x (torch.tensor): the batch to feed the network with.

        Returns:
            (torch.tensor): the predictions on the batch.
        """

        B, _, _ = x.shape

        # 2 convolutions + 1 spatial max pooling
        x = self.bn1(x)  # (B, C, V)
        x = self.conv1(x)  # (B, C, V)
        x = self.bn2(x)  # (B, C, V)
        x = self.conv2(x)  # (B, C, V)
        x = x.view(B, -1, self.nx3[0], self.nx2[0], self.nx1[0])  # (B, C, L, H, W)
        x = self.pooling(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, C, L, H', W')
        x = x.view(B, -1, self.nx3[1] * self.nx2[1] * self.nx1[1])  # (B, C, V)

        # 2 convolutions + 1 spatial max pooling
        x = self.bn3(x)  # (B, C, V)
        x = self.conv3(x)  # (B, C, V)
        x = self.bn4(x)  # (B, C, V)
        x = self.conv4(x)  # (B, C, V)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, self.nx3[1], self.nx2[1], self.nx1[1])  # (B, C, L, H, W)
        x = self.pooling(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, C, L, H', W')
        x = x.view(B, -1, self.nx3[2] * self.nx2[2] * self.nx1[2])  # (B, C, V)

        # 2 convolutions + 1 global max pooling
        x = self.bn5(x)  # (B, C, V)
        x = self.conv5(x)  # (B, C, V)
        x = self.bn6(x)  # (B, C, V)
        x = self.conv6(x)  # (B, C, V)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, self.nx3[2], self.nx2[2], self.nx1[2])  # (B, C, L, H, W)
        x = self.pooling(x, kernel_size=(self.nx3[2], self.nx2[2], self.nx1[2])).squeeze()  # (B, C)

        x = F.log_softmax(x, dim=1)  # (B, C)

        return x

    @property
    def capacity(self):
        """
        Return the capacity of the network, i.e. its number of trainable parameters.

        Returns:
            (int): the number of parameters of the network.
        """
        return sum(p.numel() for p in self.parameters())
