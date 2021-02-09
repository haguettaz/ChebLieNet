from typing import Optional, Tuple

import torch
import wandb
from ignite.engine.engine import Engine
from torch import FloatTensor, Tensor
from torch import device as Device
from torch.utils.data import DataLoader

from ..graph.graph import Graph


def prepare_batch(
    batch: Tuple[Tensor, Tensor], graph: Graph, device: Device
) -> Tuple[Tensor, Tensor]:
    """
    Prepares a batch to directly feed a model with.

    Args:
        batch (tuple): batch containing input and target data.
        graph (Graph): graph object corresponding to the support of the input.
        device (Device): device on which to put the batch data.

    Returns:
        (Tensor): batch input.
        (Tensor): batch target.
    """
    input, target = batch
    B, C, H, W = input.shape

    input = input.unsqueeze(2).expand(B, C, graph.nsym, H, W)
    # input = graph.project(input)
    input = input.reshape(B, C, -1)  # (B, C, L*H*W)

    return input.to(device), target.to(device)


def wandb_log(trainer: Engine, evaluator: Engine, data_loader: DataLoader):
    """
    Evaluate a model and add performance to wandb.

    Args:
        trainer (Engine): trainer engine.
        evaluator (Engine): evaluator engine.
        data_loader (DataLoader): dataloader on which to evaluate the model.
    """
    evaluator.run(data_loader)
    metrics = evaluator.state.metrics

    for k in metrics:
        wandb.log({k: metrics[k], "epoch": trainer.state.epoch})


def set_sparse_laplacian(trainer: Engine, graph: Graph, on: str, rate: float, device: Device):
    graph.set_sparse_laplacian(on, rate, norm=True, device=device)
    wandb.log({f"{on} sparsification": rate, "epoch": trainer.state.epoch})
