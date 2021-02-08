import torch
from gechebnet.graph.graph import Graph, SE2GEGraph, SO3GEGraph
from gechebnet.model.chebnet import GEChebNet
from gechebnet.model.reschebnet import ResGEChebNet
from torch import device as Device
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.sparse import FloatTensor as SparseFloatTensor


def get_graph(
    lie_group: str, dataset: str, nsym: int, knn: int, eps: float, xi: float, device: Device
) -> Graph:
    """
    [summary]

    Args:
        lie_group (str): [description]
        dataset (str): [description]
        nsym (int): [description]
        knn (int): [description]
        eps (float): [description]
        xi (float): [description]

    Raises:
        ValueError: [description]

    Returns:
        Graph: [description]
    """
    if lie_group == "se2":
        graph = SE2GEGraph(
            nx=28 if dataset == "mnist" else 96,
            ny=28 if dataset == "mnist" else 96,
            ntheta=nsym,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_kernel=lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac),
            device=device,
        )

    elif lie_group == "so3":
        graph = SO3GEGraph(
            nsamples=28 * 28 if dataset == "mnist" else 96 * 96,
            nalpha=nsym,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_kernel=lambda sqdistc, sigmac: torch.exp(-sqdistc / sigmac),
            device=device,
        )

    return graph


def get_model(
    graph: Graph,
    in_channels: int,
    hidden_channels: list,
    out_channels: int,
    K: int,
    pooling: str,
    resnet: bool,
    device: Device,
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
            hidden_channels,
            out_channels,
            pooling,
            device,
        )
    else:
        model = GEChebNet(
            graph,
            K,
            in_channels,
            hidden_channels[0],
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
