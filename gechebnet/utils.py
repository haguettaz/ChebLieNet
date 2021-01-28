import math
from typing import Optional, Union

import numpy as np
import torch
from numpy import ndarray
from pykeops.torch import LazyTensor
from scipy.sparse import coo_matrix
from torch import Tensor
from torch.sparse import FloatTensor as SparseFloatTensor
from torch.sparse import Tensor as SparseTensor
from torch import device as Device

def rescale(input: Tensor, low: Union[int, float] = 0.0, up: Union[int, float] = 1.0) -> Tensor:
    """
    Returns a new tensor with the rescaled version of the elements of input.

    Args:
        input (Tensor): input tensor.
        low (int or float, optional): lowest value of output.
        up (int or float, optional): highest value of output.

    Returns:
        (Tensor): rescaled input.
    """

    max_, _ = torch.max(input, dim=-1)
    min_, _ = torch.min(input, dim=-1)

    if max_ == min_:
        return (input - min_) + low

    return (up - low) * torch.divide(input - min_, max_ - min_) + low


def mod(input: Tensor, n: float, d: float = 0.0) -> Tensor:
    """
    Returns a new tensor with the modulo with offset of the elements of input.

    Args:
        input (Tensor): input tensor.
        n (float): modulus.
        d (float, optional): offset. Defaults to 0.0.

    Returns:
        (Tensor): output tensor.
    """
    return (input - d) % n + d


def shuffle_tensor(input: Tensor) -> Tensor:
    """
    Returns a new tensor with a shuffling of the elements of input.

    Args:
        input (Tensor): input tensor.

    Returns:
        (Tensor): output tensor.
    """
    return input[torch.randperm(input.nelement())]


def random_choice(input: Tensor) -> Tensor:
    """
    Returns a random element of input.

    Args:
        input (Tensor): input tensor.

    Returns:
        (Tensor): output tensor.
    """
    return Tensor[torch.randint(input.nelement(), (1,))]


def sparsity_measure(input: SparseTensor) -> float:
    """
    Returns the sparsity rate of the input sparse tensor, i.e. percentage of zero elements.

    Args:
        input (SparseTensor): input tensor.

    Returns:
        (float): sparsity rate.
    """
    return input._nnz() / (input.size(0) * input.size(1))


def sparse_tensor_to_sparse_array(input) -> coo_matrix:
    """
    Returns a sparse version of the input tensor.

    Args:
        input (SparseTensor): sparse tensor.

    Returns:
        (coo_matrix): output matrix.
    """
    input = input.cpu()

    row, col = input._indices()
    value = input._values()

    out = coo_matrix((value, (row, col)), input.size())
    return out


def sparse_tensor_diag(
    size: int, diag: Tensor = None, device: Device = None
) -> SparseFloatTensor:
    """
    Returns a diagonal sparse tensor.

    Args:
        size (int): number of diagonal elements.
        diag (Tensor, optional): elements of the diagonal. Defaults to None.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        (SparseFloatTensor): output tensor.
    """

    device = device or Device("cpu")
    diag = diag or torch.ones(size)

    return SparseFloatTensor(
        indices=torch.arange(size).expand(2, -1), values=diag, size=(size, size), device=device
    )
