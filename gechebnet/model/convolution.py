"""Chebyshev convolution layer. For the moment taking as-is from Michaël Defferrard's implementation. For v0.15 we will rewrite parts of this layer.
"""

import math

import torch
from torch import FloatTensor
from torch import device as Device
from torch.nn import Module, Parameter
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..utils import sparse_tensor_diag


def cheb_conv(x: FloatTensor, laplacian: SparseFloatTensor, weight: FloatTensor) -> FloatTensor:
    """
    Chebyshev convolution.

    Args:
        x (FloatTensor): data input in format (B, Cin, V).
        laplacian (SparseFloatTensor): symmetric normalized laplacian.
        weight (FloatTensor): layer's weights in format (K, Cin, Cout).

    Returns:
        (FloatTensor): convolved data input in format (V, B, Cout).
    """

    B, Cin, V = x.shape  # (B, Cin, V)
    K, _, _ = weight.shape  # (K, Cin, Cout)

    x0 = x.permute(2, 0, 1).contiguous().view(V, B * Cin)  # (B, Cin, V) -> (V, B*Cin)
    x = x0.unsqueeze(0)  # (V, B*Cin) -> (1, V, B*Cin)
    print("0", x.isnan().sum())
    print(x0.min(), x0.max())
    if K > 1:
        x1 = torch.mm(laplacian, x0)  # (V, B*Cin)
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # (1, V, B*Cin) -> (2, V, B*Cin)
        print("1", x.isnan().sum())
        print(x1.min(), x1.max())

        for k in range(2, K):
            x2 = 2 * torch.mm(laplacian, x1) - x0  # -> (V, B*Cin)
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # (k-1, V, B*Cin) -> (k, V, B*Cin)
            print(k, x.isnan().sum())
            print(x2.min(), x2.max())
            x0, x1 = x1, x2  # (V, B*Cin), (V, B*Cin)

    x = x.contiguous().view(K, V, B, Cin)  # (K, V, B*Cin) -> (K, V, B, Cin)
    x = torch.tensordot(x, weight, dims=([0, 3], [0, 1]))  # (V, B, Cout)

    return x


class ChebConv(Module):
    """Graph convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias=True,
    ):
        """
        Initialize the Chebyshev layer.

        Args:
            in_channels (int): number of channels in the input graph.
            out_channels (int): number of channels in the output graph.
            kernel_size (int): number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool, optional): whether to add a bias term. Defaults to True.
        """
        super().__init__()

        if kernel_size < 1:
            raise ValueError(
                f"{kernel_size} is not a valid value for kernel_size: must be strictly positive"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        shape = (kernel_size, in_channels, out_channels)
        self.weight = Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._kaiming_initialization()

    def _kaiming_initialization(self):
        """Initialize weights and bias."""
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, x: FloatTensor, laplacian: SparseFloatTensor):
        """Forward graph convolution.

        Args:
            x (FloatTensor): input data.

        Returns:
            (FloatTensor): convolved input data.
        """
        print("before", x.isnan().sum())
        x = cheb_conv(x, laplacian, self.weight)  # (B, V, Cin) -> (V, B, Cout)
        print("after", x.isnan().sum())

        if self.bias is not None:
            x += self.bias  # (V, B, Cout) -> (V, B, Cout)

        x = x.permute(1, 2, 0).contiguous()  # (B, Cout, V)
        return x
