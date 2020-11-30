import torch
import wandb


def prepare_batch(batch, device, non_blocking):
    """
    Prepare the batch and return freshly baked inputs and targets to feed the model with.

    Args:
        batch (Batch): the batch.
        device (torch.device): the device to put tensors on.
        non_blocking (bool): ...

    Returns:
        (torch.tensor): the input.
        (torch.tensor): the target.
    """
    device = device or torch.device("cpu")
    x = batch.to(device)
    y = batch.y.to(device)
    return x, y


def track_loss(trainer):
    """
    [summary]

    Args:
        trainer ([type]): [description]
    """
    wandb.log({"iteration": trainer.state.iteration, "loss": trainer.state.output})


def track_metrics(trainer, evaluator, data_loader, name):
    """
    [summary]

    Args:
        trainer ([type]): [description]
        evaluator ([type]): [description]
        data_loader ([type]): [description]
        name ([type]): the name of the dataset where the model is evaluated.
    """
    evaluator.run(data_loader)
    metrics = evaluator.state.metrics
    wandb.log({f"{k} on {name} set": metrics[k] for k in metrics})


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
