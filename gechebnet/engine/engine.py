from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from ignite.engine.engine import Engine
from ignite.metrics import Metric


def create_supervised_trainer(
    nx: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[torch.device] = None,
    prepare_batch: Callable = None,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
) -> Engine:
    """Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.
        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_

        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    Returns:
        Engine: a trainer engine with supervised update function.
    """

    device = device or torch.device("cpu")

    if prepare_batch is None:
        raise ValueError("prepare_batch function must be specified")

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, nx, device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        return output_transform(x, y, y_pred, loss)

    trainer = Engine(_update)

    return trainer


def create_supervised_evaluator(
    nx: tuple,
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[torch.device] = None,
    prepare_batch: Callable = None,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note:
        `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_

        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """

    device = device or torch.device("cpu")

    metrics = metrics or {}

    if prepare_batch is None:
        raise ValueError("prepare_batch function must be specified")

    def _inference(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, nx, device)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
