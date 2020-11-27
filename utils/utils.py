import torch


def shuffle(tensor):
    """
    Shuffle a torch tensor.

    Args:
        tensor (torch.tensor): a tensor

    Returns:
        (torch.tensor): a permuted version of the original tensor
    """
    return tensor[torch.randperm(tensor.nelement())]


def random_choice(tensor):
    """
    [summary]

    Args:
        tensor ([type]): [description]

    Returns:
        [type]: [description]
    """
    return tensor[torch.randint(tensor.nelement(), (1,))]
