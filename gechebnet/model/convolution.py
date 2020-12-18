"""Chebyshev convolution layer. For the moment taking as-is from Michaël Defferrard's implementation. For v0.15 we will rewrite parts of this layer.
"""

import math

import torch
from torch import nn

from ..utils import sparse_tensor_diag


def cheb_conv(laplacian, x, weight):
    """Chebyshev convolution.
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        x (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """

    # B = batch size
    # V = nb vertices
    # Cin = nb input channels
    # Cout = nb output channels
    # K = order of Chebyshev polynomials

    B, Cin, V = x.shape  # (B, Cin, V)
    K, _, _ = weight.shape  # (K, Cin, Cout)

    x0 = x.permute(2, 0, 1).contiguous().view(V, B * Cin)  # (B, Cin, V) -> (V, B*Cin)

    x = x0.unsqueeze(0)  # (V, B*Cin) -> (1, V, B*Cin)

    if K > 1:
        x1 = torch.mm(laplacian, x0)  # (V, B*Cin)
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # (1, V, B*Cin) -> (2, V, B*Cin)
        for _ in range(1, K - 1):
            x2 = torch.addmm(x0, laplacian, x1, beta=-1, alpha=2)  # -> (V, B*Cin)
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # (k-1, V, B*Cin) -> (k, V, B*Cin)
            x0, x1 = x1, x2  # (V, B*Cin), (V, B*Cin)

    x = x.contiguous().view(K, V, B, Cin)  # (K, V, B*Cin) -> (K, V, B, Cin)
    x = torch.tensordot(x, weight, dims=([0, 3], [0, 1]))  # (V, B, Cout)

    return x


class ChebConv(torch.nn.Module):
    """Graph convolutional layer."""

    def __init__(self, graph, in_channels, out_channels, kernel_size, bias=True, conv=cheb_conv, laplacian_device=None):
        """Initialize the Chebyshev layer.
        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
            conv (callable): Function which will perform the actual convolution.
        """
        super().__init__()
        laplacian_device = laplacian_device or torch.device("cpu")

        if kernel_size < 1:
            raise ValueError(f"{kernel_size} is not a valid value for kernel_size: must be strictly positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv

        shape = (kernel_size, in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._kaiming_initialization()

        self.laplacian = self._norm(graph.laplacian, graph.lmax, graph.num_nodes).to(laplacian_device)

    def _norm(self, laplacian, lmax, num_nodes):
        """Scale the eigenvalues from [0, lmax] to [-1, 1]."""
        return 2 * laplacian / lmax - sparse_tensor_diag(num_nodes)

    def _kaiming_initialization(self):
        """Initialize weights and bias."""
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, x):
        """Forward graph convolution.
        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            x (:obj:`torch.Tensor`): The current input data being forwarded.
        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        x = self._conv(self.laplacian, x, self.weight)

        if self.bias is not None:
            x += self.bias  # (V, B, Cout)

        x = x.permute(1, 2, 0).contiguous()  # (B, Cout, V)
        return x
