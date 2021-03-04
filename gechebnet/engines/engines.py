# coding=utf-8

import torch
from ignite.engine.engine import Engine


def create_supervised_trainer(
    graph,
    model,
    optimizer,
    loss_fn,
    device=None,
    prepare_batch=None,
    output_transform=None,
):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        graph (`Graph`): graph.
        model (`torch.nn.Module`): neural network.
        optimizer (`torch.optim.Optimizer`): optimizer.
        loss_fn (callable): loss function.
        device (`torch.device`, optional): computation device. Defaults to None.
        prepare_batch (callable, optional): function that receives `batch`, `graph` and `device` and outputs
            tuple of tensors `(batch_x, batch_y)`. Defaults to None.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default to None.

    Returns:
        (`Engine`): trainer ignite's engine with supervised update function.
    """

    device = device or torch.device("cpu")
    prepare_batch = prepare_batch if prepare_batch is not None else lambda x, y: (x, y)
    output_transform = output_transform if output_transform is not None else lambda x, y, y_pred, loss: loss.item()

    def _update(engine, batch):
        """
        Args:
            engine (`Engine`): trainer ignite's engine.
            batch (tuple): tuple of tensors `(batch_x, batch_y)`.

        Returns:
            (any): specified by output transforms.
        """
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, graph, device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    trainer = Engine(_update)

    return trainer


def create_supervised_evaluator(graph, model, metrics=None, device=None, prepare_batch=None, output_transform=None):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        graph (`Graph`): graph.
        model (`torch.nn.Module`): neural network.
        metrics (dict, optional): map metric names to Metrics. Defaults to None.
        device (`torch.device`, optional): computation device. Defaults to None.
        prepare_batch (callable, optional): function that receives `batch`, `graph` and `device` and outputs
            tuple of tensors `(batch_x, batch_y)`. Defaults to None.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default to None.

    Returns:
        (`Engine`): evaluator ignite's engine with supervised inference function.
    """

    metrics = metrics or {}

    device = device or torch.device("cpu")

    prepare_batch = prepare_batch if prepare_batch is not None else lambda x, y: (x, y)
    output_transform = output_transform if output_transform is not None else lambda x, y, y_pred, loss: (y_pred, y)

    def _inference(engine, batch):
        """
        Args:
            engine (`Engine`): evaluator engine.
            batch (tuple): tuple of tensors `(batch_x, batch_y)`.

        Returns:
            (any): specified by output transforms.
        """
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, graph, device)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
