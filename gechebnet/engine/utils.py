# pylint: disable=E1101

from typing import Optional, Tuple

import torch
import wandb
from ignite.engine.engine import Engine
from torch import FloatTensor, Tensor
from torch import device as Device
from torch.utils.data import DataLoader

from ..graph.graph import Graph


def prepare_batch(batch: Tuple[Tensor, Tensor], graph: Graph, device: Device) -> Tuple[Tensor, Tensor]:
    """
    Prepares a batch to directly feed a model with.

    Args:
        batch (tuple): batch containing input and target data.
        graph (Graph): graph.
        device (Device): computation device.

    Returns:
        (Tensor): batch input.
        (Tensor): batch target.
    """
    input, target = batch
    B, C, H, W = input.shape

    input = input.unsqueeze(2).expand(B, C, graph.nx3, H, W)
    input = input.reshape(B, C, -1)  # (B, C, L*H*W)

    input = graph.project(input)

    return input.to(device), target.to(device)


def wandb_log(trainer: Engine, evaluator: Engine, data_loader: DataLoader):
    """
    Evaluates a model and log information with wandb.

    Args:
        trainer (Engine): trainer engine.
        evaluator (Engine): evaluator engine.
        data_loader (DataLoader): dataloader on which to evaluate the model.
    """
    evaluator.run(data_loader)
    metrics = evaluator.state.metrics
    for k in metrics:
        wandb.log({k: metrics[k], "epoch": trainer.state.epoch})


def edges_dropout(trainer: Engine, graph: Graph, rate: float):
    """
    Random sampling on edges of the graph.

    Args:
        trainer (Engine): trainer engine.
        graph (Graph): graph.
        rate (float): rate of edges to randomly sample.
    """
    if hasattr(graph, "laplacian"):
        del graph.laplacian

    graph.edge_sampling(rate)
    wandb.log({"num_nodes": graph.num_nodes, "epoch": trainer.state.epoch})
    wandb.log({"num_edges": graph.num_edges, "epoch": trainer.state.epoch})


def nodes_sparsification(trainer: Engine, graph: Graph, rate: float):
    """
    Random sampling on nodes of the graph.

    Args:
        trainer (Engine): trainer engine.
        graph (Graph): graph.
        rate (float): rate of nodes to randomly sample.
    """
    if hasattr(graph, "laplacian"):
        del graph.laplacian

    if hasattr(graph, "node_proj"):
        del graph.node_proj

    graph.node_sampling(rate)
    wandb.log({"num_nodes": graph.num_nodes, "epoch": trainer.state.epoch})
    wandb.log({"num_edges": graph.num_edges, "epoch": trainer.state.epoch})
