from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from ignite.engine.engine import Engine
from ignite.metrics import Metric
from torch import Tensor
from torch import device as Device
from torch.nn import Module
from torch.optim import Optimizer

from ..graph.graph import Graph


def create_supervised_trainer(
    graph: Graph,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    device: Optional[Device] = None,
    prepare_batch: Optional[Callable] = None,
    output_transform: Optional[Callable] = lambda x, y, y_pred, loss: loss.item(),
) -> Engine:
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (Module): neural network.
        optimizer (Optimizer): optimizer.
        loss_fn (callable): loss function.
        device (Device, optional): computation device. Defaults to None.
        prepare_batch (callable, optional): function that receives `batch`, `graph` and `device` and outputs
            tuple of tensors `(batch_x, batch_y)`. Defaults to None.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default to lambda x, y, y_pred, loss: loss.item().

    Raises:
        ValueError: prepare batch function has to be defined.

    Returns:
        (Engine): trainer engine with supervised update function.
    """

    device = device or Device("cpu")

    if prepare_batch is None:
        raise ValueError("prepare_batch function must be specified")

    def _update(engine: Engine, batch: Tuple[Tensor, Tensor]) -> Any:
        """
        Updates engine.

        Args:
            engine (Engine): trainer engine.
            batch (tuple): tuple of tensors `(batch_x, batch_y)`.

        Returns:
            Any: specified by output transforms.
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


def create_supervised_evaluator(
    graph: Graph,
    model: Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Device] = None,
    prepare_batch: Callable = None,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (Module): neural network.
        metrics (dict, optional): map metric names to Metrics. Defaults to None.
        device (Device): computation device. Defaults to None.
        prepare_batch (callable, optional): function that receives `batch`, `graph` and `device` and outputs
            tuple of tensors `(batch_x, batch_y)`. Defaults to None.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default to lambda x, y, y_pred, loss: (y_pred, y).

    Raises:
        ValueError("prepare_batch function must be specified")

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """

    device = device or Device("cpu")

    metrics = metrics or {}

    if prepare_batch is None:
        raise ValueError("prepare_batch function must be specified")

    def _inference(engine: Engine, batch: Tuple[Tensor, Tensor]) -> Any:
        """
        Infers

        Args:
            engine (Engine): evaluator engine.
            batch (tuple): tuple of tensors `(batch_x, batch_y)`.

        Returns:
            Any: specified by output transforms.
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
