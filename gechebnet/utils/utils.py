import math

import numpy as np
import scipy
import torch


def mod(input, n, d=0.0):
    """
    Returns a new tensor whose elements corresepond to the modulo with offset of the elements of the input.

    Args:
        input (`torch.FloatTensor`): input tensor.
        n (float): modulus.
        d (float, optional): offset. Defaults to 0.0.

    Returns:
        (`torch.FloatTensor`): output tensor.
    """
    return (input - d) % n + d


def sinc(input):
    """
    Returns a new tensor whose elements correspond to the sinus cardinal of the elements of the input.

    Args:
        input (`torch.FloatTensor`): input tensor.

    Returns:
        (`torch.FloatTensor`): output tensor.
    """
    output = torch.sin(input) / input
    output[input == 0.0] = 1.0
    return output


def round(input, n_digits=0):
    """
    Returns a new tensor with the rounded to n decimal places version of the elements of the input.

    Args:
        input (`torch.FloatTensor`): input_tensor.
        n_digits (int, optional): number of digits. Defaults to 0.

    Returns:
        (`torch.FloatTensor`): output tensor.
    """
    return torch.round(input * 10 ** n_digits) / (10 ** n_digits)


def weighted_norm(input, Re):
    """
    Returns a new tensor whose elements correspond to the squared weighted norm of the input.

    Args:
        input (`torch.FloatTensor`): input tensor with shape (N, D).
        Re (`torch.FloatTensor`): metric tensor with shape (D, D).

    Returns:
        (`torch.FloatTensor`): squared weighted norm with shape (N).
    """
    Re = Re.to(input.device)
    return torch.matmul(torch.matmul(input.transpose(1, 2), Re), input).squeeze()


def shuffle_tensor(input):
    """
    Returns a new tensor whose elements correspond to a randomly shuffled version of the the elements of the input.

    Args:
        input (`torch.Tensor`): input tensor.

    Returns:
        (`torch.Tensor`): output tensor.
    """
    return input[torch.randperm(input.nelement())]


def sparsity_measure(input):
    """
    Returns the sparsity rate of the input sparse tensor, i.e. the rate of zero elements.

    Args:
        input (`torch.sparse.Tensor`): input tensor.

    Returns:
        (float): sparsity rate.
    """
    return input._nnz() / (input.size(0) * input.size(1))


def sparse_tensor_to_sparse_array(input):
    """
    Converts a sparse torch tensor to a sparse scipy matrix.

    Args:
        input (`torch.sparse.Tensor`): torch tensor.

    Returns:
        (`scipy.sparse.coo_matrix`): scipy matrix.
    """
    input = input.cpu()

    row, col = input._indices()
    value = input._values()

    out = scipy.sparse.coo_matrix((value, (row, col)), input.size())
    return out


def delta_kronecker(size, offset, device=None):
    """
    Returns a new tensor whose elements correspond to a delta kronecker with offset.

    Args:
        size (int or tuple of ints): size of the output tensor.
        offset (int or tuple of ints): offset of the delta kronecker.
        device (`torch.device`, optional): computation device. Defaults to None.

    Returns:
        (`torch.FloatTensor`): delta kronecker.
    """

    if type(size) != type(offset):
        raise ValueError(f"size and offset must have the same type: get {type(size)} and {type(offset)}.")

    if isinstance(size, tuple) and len(size) != len(offset):
        raise ValueError(f"size and offset must have the same length: get {len(size)} and {len(offset)}.")

    output = torch.zeros(size, device=device)
    output[offset] = 1.0
    return output
