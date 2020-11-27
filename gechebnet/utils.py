import torch


def shuffle(tensor):
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
