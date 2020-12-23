import scipy
import torch


def rect(x, a, b):
    # a is inclusive, b is exclusive
    eps = 1e-5
    return (-x + b - eps).step() * (x - a).step()


def lower(x, b, inclusive=True):
    if not inclusive:
        return (-x - 1e-4 + b).step()
    return (-x + b).step()


def upper(x, a, inclusive=True):
    if not inclusive:
        return (x - 1e-5 - a).step()
    return (x - a).step()


def mod(x, divider, offset):
    return (
        rect(x, offset - divider, offset) * (x + 1.5 * divider + offset)
        + rect(x, offset, offset + divider) * (x + 0.5 * divider + offset)
        + rect(x, offset + divider, offset + 2 * divider) * (x - 0.5 * divider + offset)
    )


def normalize(signal):
    max_, _ = torch.max(signal, dim=0)
    min_, _ = torch.min(signal, dim=0)

    return torch.divide(signal - min_, max_ - min_)


def shuffle_tensor(tensor):
    """
    Randomly permute elements of a tensor.

    Args:
        tensor (torch.tensor): the tensor

    Returns:
        (torch.tensor): the shuffled tensor.
    """
    return tensor[torch.randperm(tensor.nelement())]


def random_choice(tensor):
    """
    Randomly pick one element of a tensor.

    Args:
        tensor (torch.tensor): the tensor

    Returns:
        (torch.tensor): the picked element.
    """
    return tensor[torch.randint(tensor.nelement(), (1,))]


def sparsity_measure(sparse_tensor):
    return sparse_tensor._nnz() / (sparse_tensor.size(0) * sparse_tensor.size(1))


def sparse_tensor_to_sparse_array(sparse_tensor):
    sparse_tensor = sparse_tensor.cpu()

    row, col = sparse_tensor._indices()
    value = sparse_tensor._values()

    out = scipy.sparse.coo_matrix((value, (row, col)), sparse_tensor.size())
    return out


def sparse_tensor_diag(n, diag=None, device=None):
    diag = diag or torch.ones(n)

    return torch.sparse.FloatTensor(torch.arange(n).expand(2, -1), diag, torch.Size((n, n))).to(device)
