import argparse
import math
import os

import torch
import wandb
from gechebnet.data.dataloader import get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, wandb_log
from gechebnet.graph.graph import Graph, SE2GEGraph, SO3GEGraph
from gechebnet.model.chebnet import GEChebNet
from gechebnet.model.reschebnet import ResGEChebNet
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch import device as Device
from torch.nn import Module
from torch.nn.functional import nll_loss
from torch.optim import Adam, Optimizer

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda")


def build_sweep_config(anisotropic: bool, coupled_sym: bool, resnet: bool, dataset: str) -> dict:
    """
    [summary]

    Args:
        anisotropic (bool): [description]
        linked (bool): [description]
        resnet (bool): [description]
        dataset (str): [description]

    Returns:
        dict: [description]
    """
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "validation_accuracy", "goal": "maximize"},
    }

    parameters = {
        "batch_size": {
            "distribution": "q_log_uniform",
            "min": math.log(8),
            "max": math.log(256) if dataset == "mnist" else math.log(64),
        },
        "K": {
            "distribution": "q_log_uniform",
            "min": math.log(2),
            "max": math.log(16) if dataset == "mnist" else math.log(32),
        },
        "knn": {"distribution": "categorical", "values": [4, 8, 16, 32]},
        "learning_rate": {
            "distribution": "log_uniform",
            "min": math.log(1e-5),
            "max": math.log(0.1),
        },
        "pooling": {"distribution": "categorical", "values": ["avg", "max"]},
        "weight_decay": {
            "distribution": "log_uniform",
            "min": math.log(1e-6),
            "max": math.log(1e-3),
        },
    }

    if anisotropic:
        if coupled_sym:
            parameters["xi"] = {
                "distribution": "log_uniform",
                "min": math.log(1e-2),
                "max": math.log(1.0),
            }
        else:
            parameters["xi"] = {"distribution": "constant", "value": 1e-4}
        parameters["nsym"] = {"distribution": "int_uniform", "min": 3, "max": 12}
        parameters["eps"] = {"distribution": "constant", "value": 0.1}

    else:
        parameters["xi"] = {"distribution": "constant", "value": 1.0}
        parameters["nsym"] = {"distribution": "constant", "value": 1}
        parameters["eps"] = {"distribution": "constant", "value": 1.0}

    sweep_config["parameters"] = parameters

    return sweep_config


def get_graph(lie_group: str, dataset: str, nsym: int, knn: int, eps: float, xi: float) -> Graph:
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
            device=DEVICE,
        )

    elif lie_group == "so3":
        graph = SO3GEGraph(
            nsamples=28 * 28 if dataset == "mnist" else 96 * 96,
            nalpha=nsym,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_kernel=lambda sqdistc, sigmac: torch.exp(-sqdistc / sigmac),
            device=DEVICE,
        )

    return graph


def get_model(
    graph: Graph,
    in_channels: int,
    hidden_channels: list,
    out_channels: int,
    K: int,
    pooling: str,
    resnet: bool = False,
    device: Device = None,
) -> Module:
    """
    [summary]

    Args:
        in_channels (int): [description]
        hidden_channels (list): [description]
        out_channels (int): [description]
        K (int): [description]
        pooling (str): [description]
        resnet (bool, optional): [description]. Defaults to False.
        device (Device, optional): [description]. Defaults to None.

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


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config

        graph = get_graph(
            lie_group=args.lie_group,
            dataset=args.dataset,
            nsym=config.nsym,
            knn=config.knn,
            eps=config.eps,
            xi=config.xi,
        )
        wandb.log({f"num_nodes": graph.num_nodes, f"num_edges": graph.num_edges})

        model = get_model(
            graph=graph,
            in_channels=1 if args.dataset == "mnist" else 3,
            hidden_channels=args.hidden_channels,
            out_channels=10,
            K=config.K,
            pooling=config.pooling,
            resnet=args.model[0] == "resnet",
            device=DEVICE,
        )
        wandb.log({"resnet": args.model[0] == "resnet"})
        wandb.log({"model_type": args.model[1]})
        wandb.log({"capacity": model.capacity})

        optimizer = get_optimizer(model, config.learning_rate, config.weight_decay)

        # Trainer and evaluator(s) engines
        trainer = create_supervised_trainer(
            graph=graph,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        metrics = {"validation_accuracy": Accuracy(), "validation_loss": Loss(nll_loss)}

        evaluator = create_supervised_evaluator(
            graph=graph,
            model=model,
            metrics=metrics,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        train_loader, val_loader = get_train_val_dataloaders(
            args.dataset,
            batch_size=config.batch_size,
            val_ratio=0.1,
            data_path=DATA_PATH,
        )

        # Performance tracking with wandb
        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experiments", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--model_type",
        nargs="+",
        type=str,
        help="Type of model: 1) isotropic or anisotropic 2) coupled_sym or uncoupled_sym 3) resnet or classicnet",
    )
    parser.add_argument("--hidden_channels", nargs="+", type=int)
    parser.add_argument("--lie_group", type=str)
    args = parser.parse_args()

    sweep_config = build_sweep_config(
        anisotropic=args.model_type[0] == "anisotropic",
        coupled_sym=args.model_type[1] == "coupled_sym",
        resnet=args.model_type[2] == "resnet",
        dataset=args.dataset,
    )

    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=args.num_experiments)
