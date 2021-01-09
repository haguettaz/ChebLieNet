from typing import Optional, Tuple

import torch.tensor
import wandb
from ignite.engine.engine import Engine
from torch.utils.data import DataLoader


def prepare_batch(batch: tuple, L: int, device: torch.device) -> Tuple[torch.tensor]:
    """
    Prepare a batch to directly feed a model with.

    Args:
        batch (tuple): batch containing input and target data.
        L (int): number of equivariance layers to use.
        device (torch.device): device on which to put the batch data.

    Returns:
        Tuple[torch.tensor]: input and target to feed a model with.
    """
    x, y = batch
    B, C, H, W = x.shape  # (B, C, H, W)

    x = x.unsqueeze(2).expand(B, C, L, H, W).reshape(B, C, -1)  # (B, C, L*H*W)

    return x.to(device), y.to(device)


def wandb_log(trainer: Engine, evaluator: Engine, data_loader: DataLoader):
    """
    Evaluate a model and add performance to wandb.

    Args:
        trainer (Engine): training engine.
        evaluator (Engine): evaluator engine.
        data_loader (DataLoader): dataloader on which to evaluate the model.
    """
    evaluator.run(data_loader)
    metrics = evaluator.state.metrics

    for k in metrics:
        wandb.log({k: metrics[k], "epoch": trainer.state.epoch})
