# coding=utf-8

import wandb


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


def output_transform(batch, cl):
    """
    output transforms for segmentation task and evaluation per class
    """
    B, C, V = y_pred.shape
    y_pred, y = batch

    y_pred = y_pred.permute(0,2,1)
    y = y.permute(0,2,1)

    mask = y == cl
    y[mask] = 1
    y[~mask] = 0

    mask = y_pred == cl
    y_pred[mask] = 1
    y_pred[~mask] = 0

    return y_pred.view(B*V, C), y.view(B*V, C)


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
