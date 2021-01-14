from typing import Optional, Union

import numpy as np
import torch
from numpy import ndarray
from pykeops.torch import LazyTensor
from scipy.sparse import coo_matrix
from torch import Tensor
from torch.sparse import FloatTensor as SparseFloatTensor
from torch.sparse import Tensor as SparseTensor


def lower(x: LazyTensor, b: Union[float, int], inclusive: Optional[bool] = True) -> LazyTensor:
    """
    Indicator function wether x is (strictly) lower than b.

    Args:
        x (LazyTensor): input.
        b (Union[float, int]): upper bound.
        inclusive (Optional[bool], optional): False if the comparision is strict. Defaults to True.

    Returns:
        (LazyTensor): x if x is (strictly) lower than b, 0 otherwise.
    """
    if not inclusive:
        return (-x - 1e-4 + b).step()
    return (-x + b).step()


def upper(x: LazyTensor, a: Union[float, int], inclusive: Optional[bool] = True) -> LazyTensor:
    """
    Indicator function wether x is (strictly) upper than b.

    Args:
        x (LazyTensor): input.
        a (Union[float, int]): upper bound.
        inclusive (Optional[bool], optional): False if the comparision is strict. Defaults to True.

    Returns:
        (LazyTensor): x if x is (strictly) higher than a, 0 otherwise.
    """
    if not inclusive:
        return (x - 1e-5 - a).step()
    return (x - a).step()


def rescale(tensor: Tensor, low: Union[int, float] = 0.0, up: Union[int, float] = 1.0) -> Tensor:
    """
    Standardize tensor

    Args:
        tensor (Tensor): input.
        low (int or float, optional): lowest value of output.
        up (int or float, optional): highest value of output.

    Returns:
        (Tensor): rescaled input.
    """

    if len(tensor.shape) == 2:
        max_, _ = torch.max(tensor, dim=1)
        min_, _ = torch.min(tensor, dim=1)

    else:
        max_, _ = torch.max(tensor, dim=0)
        min_, _ = torch.min(tensor, dim=0)

    return (up - low) * torch.divide(tensor - min_, max_ - min_) + low


def shuffle_tensor(tensor: Tensor) -> Tensor:
    """
    Randomly permute elements of an input tensor.

    Args:
        tensor (Tensor): input.

    Returns:
        (Tensor): shuffled input.
    """
    return tensor[torch.randperm(tensor.nelement())]


def random_choice(tensor: Tensor) -> Tensor:
    """
    Randomly pick one element of an input tensor.

    Args:
        tensor (Tensor): input.

    Returns:
        (Tensor): random element in input.
    """
    return Tensor[torch.randint(tensor.nelement(), (1,))]


def sparsity_measure(sparse_tensor: SparseTensor) -> float:
    """
    Measure sparsity of a sparse tensor, i.e. percentage of zero elements.

    Args:
        sparse_tensor (SparseTensor): input.

    Returns:
        float: sparsity measure.
    """
    return sparse_tensor._nnz() / (sparse_tensor.size(0) * sparse_tensor.size(1))


def sparse_tensor_to_sparse_array(sparse_tensor) -> coo_matrix:
    """
    Convert a sparse tensor (PyTorch) to a sparse coo matrix (Scipy).

    Args:
        sparse_tensor (SparseTensor): sparse tensor.

    Returns:
        coo_matrix: sparse coo matrix.
    """
    sparse_tensor = sparse_tensor.cpu()

    row, col = sparse_tensor._indices()
    value = sparse_tensor._values()

    out = coo_matrix((value, (row, col)), sparse_tensor.size())
    return out


def sparse_tensor_diag(
    size: int, diag: Tensor = None, device: torch.device = None
) -> SparseFloatTensor:
    """
    Return a diagonal sparse tensor.

    Args:
        size (int): number of diagonal elements.
        diag (Tensor, optional): elements of the diagonal. Defaults to None.
        device (torch.device, optional): computation device. Defaults to None.

    Returns:
        SparseFloatTensor: diagonal sparse tensor.
    """

    device = device or torch.device("cpu")
    diag = diag or torch.ones(size)

    return SparseFloatTensor(
        indices=torch.arange(size).expand(2, -1), values=diag, size=(size, size), device=device
    )
