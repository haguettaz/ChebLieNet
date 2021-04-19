# coding=utf-8

import torch
import wandb
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
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


# def output_transform_accuracy(batch, cl):
#     """
#     Output transforms for pixelwise accuracy per class
#     """
#     y_pred, y = batch

#     B, C, V = y_pred.shape

#     y_pred = y_pred.permute(0, 2, 1).reshape(B * V, C)
#     y = y.reshape(B * V)

#     mask = y_pred.argmax(dim=1) == cl
#     y_pred_vec = torch.zeros(B * V)
#     y_pred_vec[mask] = 1.0

#     mask = y == cl
#     y_vec = torch.zeros(B * V)
#     y_vec[mask] = 1.0

#     return y_pred_vec, y_vec


class PerClassAccuracy(Metric):
    def __init__(self, cl, output_transform=lambda x: x, device="cpu"):
        self.cl = cl
        self._num_correct = None
        self._num_examples = None
        super(PerClassAccuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(PerClassAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        indices = torch.argmax(y_pred, dim=1)
        mask = (indices == self.cl) | (y == self.cl)
        y = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("CustomAccuracy must have at least one example before it can be computed.")
        return self._num_correct.item() / self._num_examples


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
