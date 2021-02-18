"""Chebyshev convolution layer.
"""

import math
from typing import Optional

import torch
from torch import FloatTensor, nn
from torch.sparse import FloatTensor as SparseFloatTensor

from ..graph.graph import Graph


def cheb_conv(x: FloatTensor, weights: FloatTensor, laplacian: Optional[SparseFloatTensor] = None) -> FloatTensor:
    """
    Chebyshev convolution.

    Args:
        x (FloatTensor): data input in format (B, Cin, V).
        laplacian (SparseFloatTensor): symmetric normalized laplacian.
        weights (FloatTensor): layer's weights in format (R, Cin, Cout).

    Returns:
        (FloatTensor): convolved data input in format (V, B, Cout).
    """

    B, Cin, V = x.shape  # (B, Cin, V)
    R, _, _ = weights.shape  # (R, Cin, Cout)

    if laplacian is None and R > 1:
        raise ValueError(f"Can't perform Chebyschev convolution without laplacian if R > 1")

    x0 = x.permute(2, 0, 1).contiguous().view(V, B * Cin)  # (B, Cin, V) -> (V, B*Cin)
    x = x0.unsqueeze(0)  # (V, B*Cin) -> (1, V, B*Cin)

    if R > 1:
        x1 = torch.mm(laplacian, x0)  # (V, B*Cin)
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # (1, V, B*Cin) -> (2, V, B*Cin)

        for _ in range(2, R):
            x2 = 2 * torch.mm(laplacian, x1) - x0  # -> (V, B*Cin)
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # (k-1, V, B*Cin) -> (k, V, B*Cin)
            x0, x1 = x1, x2  # (V, B*Cin), (V, B*Cin)

    x = x.contiguous().view(R, V, B, Cin)  # (R, V, B*Cin) -> (R, V, B, Cin)
    x = torch.tensordot(x, weights, dims=([0, 3], [0, 1]))  # (V, B, Cout)

    return x


class ChebConv(nn.Module):
    """Graph convolutional layer."""

    def __init__(
        self,
        graph: Graph,
        in_channels: int,
        out_channels: int,
        R: int,
        bias=True,
    ):
        """
        Initialize the Chebyshev layer.

        Args:
            in_channels (int): number of channels in the input graph.
            out_channels (int): number of channels in the output graph.
            R (int): order of the Chebyshev polynomials.
            bias (bool, optional): whether to add a bias term. Defaults to True.
        """
        super().__init__()

        if R < 1:
            raise ValueError(f"{R} is not a valid value for R: must be strictly positive")

        self.graph = graph
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.R = R

        self.weight = nn.Parameter(FloatTensor(R, in_channels, out_channels))
        nn.init.kaiming_normal_(self.weight, mode="fan_in")

        if bias:
            self.bias = nn.Parameter(FloatTensor(out_channels))
            nn.init.constant_(self.bias, 0.01)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: FloatTensor):
        """Forward graph convolution.

        Args:
            x (FloatTensor): input data.

        Returns:
            (FloatTensor): convolved input data.
        """
        laplacian = self.graph.get_laplacian(norm=True, device=x.device)
        x = cheb_conv(x, self.weight, laplacian)  # (B, V, Cin) -> (V, B, Cout)

        if self.bias is not None:
            x += self.bias  # (V, B, Cout) -> (V, B, Cout)

        x = x.permute(1, 2, 0).contiguous()  # (B, Cout, V)
        return x

    def extra_repr(self) -> str:
        s = "in_channels={in_channels}, out_channels={out_channels}, R={R}"
        if self.bias is None:
            s += ", bias=False"
        else:
            s += ", bias=True"

        return s.format(**self.__dict__)
