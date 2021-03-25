import argparse
import math
import os

import torch
import wandb
from gechebnet.datas.dataloaders import get_train_val_loaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engines.engines import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engines.utils import prepare_batch, wandb_log
from gechebnet.graphs.graphs import R2GEGraph, SE2GEGraph
from gechebnet.nn.layers.pools import SE2SpatialPool
from gechebnet.nn.models.chebnets import WideResSE2GEChebNet
from gechebnet.nn.models.utils import capacity
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss
from torch.optim import Adam


def build_sweep_config() -> dict:
    """
    Gets training configuration for bayesian hyper-parameters optimization.

    Args:
        anisotropic (bool): if True, uses an anisotropic graph manifold.
        coupled_sym (bool): if True, uses coupled symmetric layers.

    Returns:
        (dict): configuration dictionnary.
    """

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "validation_accuracy", "goal": "maximize"},
    }

    parameters = {
        "batch_size": {"distribution": "categorical", "values": [8, 16, 32]},
        "kernel_size": {"distribution": "categorical", "values": [2, 3, 4, 5, 6]},
        "K": {"distribution": "categorical", "values": [8, 16, 32]},
        "xi": {"distribution": "log_uniform", "min": math.log(1e-4), "max": math.log(1e-1)},
        "ntheta": {"distribution": "int_uniform", "min": 2, "max": 9},
        "eps": {"distribution": "log_uniform", "min": math.log(1e-2), "max": math.log(1)},
    }

    sweep_config["parameters"] = parameters

    return sweep_config


def train(config=None):
    """
    Bayesian hyper-parameters optimization on MNIST.

    Args:
        config (dict, optional): configuration dictionnary. Defaults to None.
    """

    # Initialize a new wandb run
    with wandb.init(config=config):

        config = wandb.config
        wandb.log({"dataset": "mnist"})
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Loads graph manifold and set normalized laplacian
        graph = SE2GEGraph(
            [28, 28, config.ntheta],
            K=config.K,
            sigmas=(1.0, config.eps, config.xi),
            path_to_graph=args.path_to_graph,
        )

        # Loads group equivariant Chebnet
        model = WideResSE2GEChebNet(
            in_channels=1,
            out_channels=10,
            kernel_size=config.kernel_size,
            graph_lvl0=graph,
            res_depth=args.res_depth,
            widen_factor=args.widen_factor,
            reduction=args.reduction if args.pool else None,
        ).to(device)

        wandb.log({"capacity": capacity(model)})

        optimizer = Adam(model.parameters(), lr=args.lr)

        # Loads data loaders
        train_loader, val_loader = get_train_val_loaders(
            "mnist",
            batch_size=config.batch_size,
            val_ratio=0.3,
            path_to_data=args.path_to_data,
        )

        # Loads engines
        trainer = create_supervised_trainer(
            graph=graph,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        metrics = {"validation_accuracy": Accuracy(), "validation_loss": Loss(nll_loss)}
        evaluator = create_supervised_evaluator(
            graph=graph,
            model=model,
            metrics=metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

        # Launchs training
        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_graph", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--num_experiments", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--res_depth", type=int, default=2)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    sweep_config = build_sweep_config()

    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=args.num_experiments)
