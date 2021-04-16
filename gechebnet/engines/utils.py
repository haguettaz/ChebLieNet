# coding=utf-8

import torch
import wandb
from torch.nn.functional import one_hot


def prepare_batch(batch, graph, device):
    """
    Prepares a batch to directly feed a model with.

    Args:
        batch (tuple): batch tuple containing the image and the target.
        graph (`Graph`): graph.
        device (`torch.device`): computation device.

    Returns:
        (`torch.Tensor`): image.
        (`torch.Tensor`): target.
    """
    image, target = batch

    if hasattr(graph, "sub_node_index"):
        return image[..., graph.sub_node_index].to(device), target.to(device)

    return image.to(device), target.to(device)


def output_transform_mAP(batch):
    """
    Output transform for mean average precision
    """
    y_pred, y = batch

    B, C, V = y_pred.shape

    y_pred = y_pred.permute(0, 2, 1).reshape(B * V, C)
    y_pred_one_hot = one_hot(y_pred.argmax(dim=1))

    y = y.reshape(B * V)
    y_one_hot = one_hot(y)

    return y_pred_one_hot, y_one_hot


def output_transform_accuracy(batch, cl):
    """
    Output transforms for pixelwise accuracy per class
    """
    y_pred, y = batch

    B, C, V = y_pred.shape

    y_pred = y_pred.permute(0, 2, 1).reshape(B * V, C)
    y = y.reshape(B * V)

    mask = y_pred.argmax(dim=1) == cl
    y_pred_vec = torch.zeros(B * V)
    y_pred_vec[mask] = 1.0

    mask = y == cl
    y_vec = torch.zeros(B * V)
    y_vec[mask] = 1.0

    return y_pred_vec, y_vec


def wandb_log(trainer, evaluator, data_loader):
    """
    Launch the evaluator ignite's engine and log performance with wandb.

    Args:
        trainer (`Engine`): trainer ignite's engine.
        evaluator (`Engine`): evaluator ignite's engine.
        data_loader (`DataLoader`): evaluation dataloader.
    """
    evaluator.run(data_loader)
    metrics = evaluator.state.metrics
    for k in metrics:
        wandb.log({k: metrics[k], "epoch": trainer.state.epoch})


def sample_edges(trainer, graph, rate):
    """
    Perform a random edges' sampling of the given graph.
    For details, we refer to `gechebnet.graphs.graphs.RandomSubGraph.edge_sampling`

    Args:
        trainer (`Engine`): trainer ignite's engine.
        graph (`Graph`): graph.
        rate (float): rate of edges to randomly sample.
    """

    graph.reinit()
    graph.edge_sampling(rate)


def sample_nodes(trainer, graph, rate):
    """
    Perform a random nodes' sampling of the given graph.
    For details, we refer to `gechebnet.graphs.graphs.RandomSubGraph.node_sampling`

    Args:
        trainer (`Engine`): trainer ignite's engine.
        graph (`Graph`): graph.
        rate (float): rate of nodes to randomly sample.
    """
    graph.reinit()
    graph.node_sampling(rate)
