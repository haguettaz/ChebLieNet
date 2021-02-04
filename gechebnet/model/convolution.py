"""Chebyshev convolution layer.
"""

import math
from typing import Optional

import torch
from torch import FloatTensor
from torch import device as Device
from torch.nn import Module, Parameter
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph
from ..utils import sparse_tensor_diag


def cheb_conv(
    x: FloatTensor, weights: FloatTensor, laplacian: Optional[SparseFloatTensor] = None
) -> FloatTensor:
    """
    Chebyshev convolution.

    Args:
        x (FloatTensor): data input in format (B, Cin, V).
        laplacian (SparseFloatTensor): symmetric normalized laplacian.
        weights (FloatTensor): layer's weights in format (K, Cin, Cout).

    Returns:
        (FloatTensor): convolved data input in format (V, B, Cout).
    """

    B, Cin, V = x.shape  # (B, Cin, V)
    K, _, _ = weights.shape  # (K, Cin, Cout)

    if laplacian is None and K > 1:
        raise ValueError(f"Can't perform Chebyschev convolution without laplacian if K > 1")

    x0 = x.permute(2, 0, 1).contiguous().view(V, B * Cin)  # (B, Cin, V) -> (V, B*Cin)
    x = x0.unsqueeze(0)  # (V, B*Cin) -> (1, V, B*Cin)

    if K > 1:
        x1 = torch.mm(laplacian, x0)  # (V, B*Cin)
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # (1, V, B*Cin) -> (2, V, B*Cin)

        for _ in range(2, K):
            x2 = 2 * torch.mm(laplacian, x1) - x0  # -> (V, B*Cin)
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # (k-1, V, B*Cin) -> (k, V, B*Cin)
            x0, x1 = x1, x2  # (V, B*Cin), (V, B*Cin)

    x = x.contiguous().view(K, V, B, Cin)  # (K, V, B*Cin) -> (K, V, B, Cin)
    x = torch.tensordot(x, weights, dims=([0, 3], [0, 1]))  # (V, B, Cout)

    return x


class ChebConv(Module):
    """Graph convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        bias=True,
    ):
        """
        Initialize the Chebyshev layer.

        Args:
            in_channels (int): number of channels in the input graph.
            out_channels (int): number of channels in the output graph.
            K (int): order of the Chebyshev polynomials.
            bias (bool, optional): whether to add a bias term. Defaults to True.
        """
        super().__init__()

        if K < 1:
            raise ValueError(f"{K} is not a valid value for K: must be strictly positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.bias = bias

        shape = (K, in_channels, out_channels)
        self.weights = Parameter(FloatTensor(*shape))

        if bias:
            self.biases = Parameter(FloatTensor(out_channels))
        else:
            self.register_parameter("biases", None)

        self._kaiming_initialization()

    def _kaiming_initialization(self):
        """Initialize weights and bias."""
        std = math.sqrt(2 / (self.in_channels * self.K))
        self.weights.data.normal_(0, std)
        if self.bias:
            self.biases.data.fill_(0.01)

    def forward(self, x: FloatTensor, laplacian: Optional[SparseFloatTensor] = None):
        """Forward graph convolution.

        Args:
            x (FloatTensor): input data.

        Returns:
            (FloatTensor): convolved input data.
        """
        x = cheb_conv(x, self.weights, laplacian)  # (B, V, Cin) -> (V, B, Cout)

        if self.bias:
            x += self.biases  # (V, B, Cout) -> (V, B, Cout)

        x = x.permute(1, 2, 0).contiguous()  # (B, Cout, V)
        return x

    def extra_repr(self) -> str:
        return (
            "in_channels={in_channels}, out_channels={out_channels}, K={K}, "
            "bias={bias}".format(**self.__dict__)
        )
