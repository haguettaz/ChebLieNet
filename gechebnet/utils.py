import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from scipy.sparse import coo_matrix
from torch import FloatTensor, Tensor
from torch import device as Device
from torch.sparse import FloatTensor as SparseFloatTensor
from torch.sparse import Tensor as SparseTensor


def rescale(input: Tensor, low: Optional[float] = 0.0, up: Optional[float] = 1.0) -> Tensor:
    """
    Returns a new tensor with the rescaled version of the elements of input.

    Args:
        input (Tensor): input tensor.
        low (float, optional): lowest value of output. Defaults to 0.0.
        up (float, optional): highest value of output. Defaults to 1.0.

    Returns:
        (Tensor): rescaled input.
    """

    max_, _ = torch.max(input, dim=-1)
    min_, _ = torch.min(input, dim=-1)

    if torch.allclose(max_, min_):
        return (input - min_) + low

    return (up - low) * torch.divide(input - min_, max_ - min_) + low


def mod(input: Tensor, n: float, d: float = 0.0) -> Tensor:
    """
    Returns a new tensor whose elements corresepond to the modulo with offset of the elements of the input tensor.

    Args:
        input (Tensor): input tensor.
        n (float): modulus.
        d (float, optional): offset. Defaults to 0.0.

    Returns:
        (Tensor): output tensor.
    """
    return (input - d) % n + d


def sinc(input: Tensor) -> Tensor:
    """
    Returns a new tensor whose elements correspond to the sinus cardinal of the elements of the input tensor.

    Args:
        input (Tensor): input tensor.

    Returns:
        (Tensor): output tensor.
    """
    output = torch.sin(input) / input
    output[input == 0.0] = 1.0
    return output


def round(input: Tensor, n_digits: int = 0) -> Tensor:
    """
    Returns a new tensor with the rounded to n decimal places version of the elements of input.

    Args:
        input (Tensor): input_tensor.
        n_digits (int, optional): number of digits. Defaults to 0.

    Returns:
        (Tensor): output tensor.
    """
    return torch.round(input * 10 ** n_digits) / (10 ** n_digits)


def weighted_norm(input: Tensor, Re: Tensor) -> Tensor:
    """
    Returns a new tensor whose elements correspond to the squared weighted norm of the input tensor.

    Args:
        input (Tensor): input tensor.
        Re (Tensor): metric tensor.

    Returns:
        (Tensor): squared weighted norm.
    """
    Re = Re.to(input.device)
    return torch.matmul(torch.matmul(input.transpose(1, 2), Re), input).squeeze()


def shuffle_tensor(input: Tensor) -> Tensor:
    """
    Returns a new tensor whose elements correspond to a randomly shuffled version of the the elements of the input tensor.

    Args:
        input (Tensor): input tensor.

    Returns:
        (Tensor): output tensor.
    """
    return input[torch.randperm(input.nelement())]


def random_choice(input: Tensor) -> Tensor:
    """
    Returns a random element of the input tensor.

    Args:
        input (Tensor): input tensor.

    Returns:
        (Tensor): output tensor.
    """
    return Tensor[torch.randint(input.nelement(), (1,))]


def sparsity_measure(input: SparseTensor) -> float:
    """
    Returns the sparsity rate of the input sparse tensor, i.e. the rate of zero elements.

    Args:
        input (SparseTensor): input tensor.

    Returns:
        (float): sparsity rate.
    """
    return input._nnz() / (input.size(0) * input.size(1))


def sparse_tensor_to_sparse_array(input) -> coo_matrix:
    """
    Converts a PyTorch sparse tensor to a Numpy sparse array.

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


def delta_kronecker(
    size: Union[int, Tuple[int, ...]], offset: Union[int, Tuple[int, ...]], device: Optional[Device] = None
) -> Tensor:
    """
    Returns a new tensor whose elements correspond to a delta kronecker with offset.

    Args:
        size (int or tuple of ints): size of the output tensor.
        offset (int or tuple of ints): offset of the delta kronecker.
        device (Device, optional): computation device. Defaults to None.

    Returns:
        (Tensor): delta kronecker.
    """

    if type(size) != type(offset):
        raise ValueError(f"size and offset must have the same type: get {type(size)} and {type(offset)}.")

    if isinstance(size, tuple) and len(size) != len(offset):
        raise ValueError(f"size and offset must have the same length: get {len(size)} and {len(offset)}.")

    output = torch.zeros(size, device=device)
    output[offset] = 1.0
    return output
