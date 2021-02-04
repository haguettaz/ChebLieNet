from torch import device as Device
from torch.nn import Module
from torch.optim import Adam, Optimizer

from ..graph.graph import Graph
from .chebnet import GEChebNet
from .reschebnet import ResGEChebNet


def get_model(
    graph: Graph,
    in_channels: int,
    hidden_channels: list,
    out_channels: int,
    K: int,
    pooling: str,
    device: Device,
    resnet: bool = False,
) -> Module:
    """
    [summary]

    Args:
        in_channels (int): [description]
        hidden_channels (list): [description]
        out_channels (int): [description]
        K (int): [description]
        pooling (str): [description]
        device (Device): [description]. Defaults to None.
        resnet (bool, optional): [description]. Defaults to False.

    Returns:
        Module: [description]
    """
    if resnet:
        model = ResGEChebNet(
            graph,
            K,
            in_channels,
            [[hc, hc, hc] for hc in hidden_channels],
            out_channels,
            pooling,
            device,
        )
    else:
        model = GEChebNet(
            graph,
            K,
            in_channels,
            hidden_channels,
            out_channels,
            pooling,
            device,
        )
    return model.to(device)


def get_optimizer(model: Module, learning_rate: float, weight_decay: float) -> Optimizer:
    """
    Get model's parameters' optimizer.

    Args:
        model (Module): model.
        learning_rate (float): learning rate.
        weight_decay (float): weight decay.

    Returns:
        Optimizer: [description]
    """
    return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
