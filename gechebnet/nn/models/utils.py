from torch import nn


def capacity(model: nn.Module) -> int:
    """
    Returns the capacity of the model, i.e. its number of trainable parameters.

    Args:
        (nn.Module): model-

    Returns:
        (int): model's capacity.
    """
    return sum(p.numel() for p in model.parameters())
