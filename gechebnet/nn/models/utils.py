# coding=utf-8


def capacity(model):
    """
    Return the capacity of the model, i.e. its number of trainable parameters.

    Args:
        (`torch.nn.Module`): model.

    Returns:
        (int): capacity of the model.
    """
    return sum(p.numel() for p in model.parameters())
