import wandb


def prepare_batch(batch, L, device):
    """
    Prepare the batch and return freshly baked inputs and targets to feed the model with.

    Args:
        batch (Batch): the batch.
        device (torch.device): the device to put tensors on.

    Returns:
        (torch.tensor): the input.
        (torch.tensor): the target.
    """
    x, y = batch
    B, C, H, W = x.shape  # (B, C, H, W)

    x = x.unsqueeze(2).expand(B, C, L, H, W).reshape(B, C, -1)  # (B, C, L*H*W)

    return x.to(device), y.to(device)


def wandb_loss(trainer):
    """
    [summary]

    Args:
        trainer ([type]): [description]
    """
    wandb.log({"iteration": trainer.state.iteration, "loss": trainer.state.output})


def wandb_log(trainer, evaluator, data_loader):
    """
    [summary]

    Args:
        trainer ([type]): [description]
        evaluator ([type]): [description]
        data_loader ([type]): [description]
    """
    evaluator.run(data_loader)
    metrics = evaluator.state.metrics

    for k in metrics:
        wandb.log({k: metrics[k], "epoch": trainer.state.epoch})
