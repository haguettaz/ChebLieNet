from torch.nn import Module
from torch.optim import SGD, Adam, Optimizer


def get_optimizer(
    model: Module, optimizer: str, learning_rate: float, weight_decay: float
) -> Optimizer:
    """
    Get model's parameters' optimizer.

    Args:
        model (Module): model.
        optimizer (str): name of optimizer.
        learning_rate (float): learning rate.
        weight_decay (float): weight decay.

    Raises:
        ValueError: optimizer must be 'sgd' or 'adam'.

    Returns:
        Optimizer: [description]
    """
    if optimizer not in {"sgd", "adam"}:
        raise ValueError(f"{optimizer} is not a valid value for pooling: must be 'sgd' or 'adam'")

    if optimizer == "sgd":
        optimizer = SGD(
            model.parameters(), lr=learning_rate * 100, momentum=0.9, weight_decay=weight_decay
        )
    elif optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer
