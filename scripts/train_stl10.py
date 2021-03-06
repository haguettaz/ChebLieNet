import argparse
import os

import torch
import wandb
from cheblienet.datas.dataloaders import get_test_loader, get_train_val_loaders
from cheblienet.engines.engines import create_supervised_evaluator, create_supervised_trainer
from cheblienet.engines.utils import prepare_batch, wandb_log
from cheblienet.graphs.graphs import R2GEGraph, SE2GEGraph
from cheblienet.nn.layers.pools import SE2SpatialPool
from cheblienet.nn.models.chebnets import WideResSE2GEChebNet
from cheblienet.nn.models.utils import capacity
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss
from torch.optim import Adam


def build_config(anisotropic: bool) -> dict:
    """
    Gets training configuration.

    Args:
        anisotropic (bool): if True, uses an anisotropic graph manifold.

    Returns:
        (dict): configuration dictionnary.
    """

    if not anisotropic:
        return {
            "kernel_size": 4,
            "eps": 1.0,
            "K": 8,
            "ntheta": 1,
            "xi_0": 1.0,
            "xi_1": 1.0,
            "xi_2": 1.0,
        }

    return {
        "kernel_size": 4,
        "eps": 0.1,
        "K": 16,
        "ntheta": 6,
        "xi_0": 2.048 / (24 ** 2),
        "xi_1": 2.048 / (48 ** 2),
        "xi_2": 2.048 / (96 ** 2),
    }


def train(config=None):
    """
    Trains a model on STL10 and evaluates its performance on STL10, Flip-STL10 and 90Rot-STL10.

    Args:
        config (dict, optional): configuration dictionnary. Defaults to None.
    """

    # Initialize a new wandb run
    with wandb.init(config=config, project="cheblienet"):

        config = wandb.config
        wandb.log({"dataset": "stl10"})
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Load model and optimizer
        if args.anisotropic:
            graph_lvl0 = SE2GEGraph(
                [24, 24, config.ntheta],
                K=config.K,
                sigmas=(1.0, config.eps, config.xi_0),
                path_to_graph=args.path_to_graph,
            )

            graph_lvl1 = SE2GEGraph(
                [48, 48, config.ntheta],
                K=config.K,
                sigmas=(1.0, config.eps, config.xi_1),
                path_to_graph=args.path_to_graph,
            )

            graph_lvl2 = SE2GEGraph(
                [96, 96, config.ntheta],
                K=config.K,
                sigmas=(1.0, config.eps, config.xi_2),
                path_to_graph=args.path_to_graph,
            )
        else:
            graph_lvl0 = R2GEGraph(
                [24, 24, 1],
                K=config.K,
                sigmas=(1.0, config.eps, config.xi_0),
                path_to_graph=args.path_to_graph,
            )

            graph_lvl1 = R2GEGraph(
                [48, 48, 1],
                K=config.K,
                sigmas=(1.0, config.eps, config.xi_1),
                path_to_graph=args.path_to_graph,
            )

            graph_lvl2 = R2GEGraph(
                [96, 96, 1],
                K=config.K,
                sigmas=(1.0, config.eps, config.xi_2),
                path_to_graph=args.path_to_graph,
            )

        # Loads group equivariant Chebnet
        model = WideResSE2GEChebNet(
            in_channels=3,
            out_channels=10,
            kernel_size=config.kernel_size,
            graph_lvl0=graph_lvl0,
            graph_lvl1=graph_lvl1,
            graph_lvl2=graph_lvl2,
            res_depth=args.res_depth,
            widen_factor=args.widen_factor,
            reduction=args.reduction,
        ).to(device)

        wandb.log({"capacity": capacity(model)})

        optimizer = Adam(model.parameters(), lr=args.lr)

        # Load dataloaders
        train_loader, _ = get_train_val_loaders(
            "stl10",
            batch_size=args.batch_size,
            val_ratio=0.0,
            num_layers=config.ntheta,
            path_to_data=args.path_to_data,
        )

        test_loader = get_test_loader(
            "stl10", batch_size=args.batch_size, num_layers=config.ntheta, path_to_data=args.path_to_data
        )

        # Load engines
        trainer = create_supervised_trainer(
            graph=graph_lvl2,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        metrics = {"test_accuracy": Accuracy(), "test_loss": Loss(nll_loss)}

        evaluator = create_supervised_evaluator(
            graph=graph_lvl2,
            model=model,
            metrics=metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, test_loader)

        # Launchs training
        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_graph", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--num_experiments", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--anisotropic", action="store_true", default=False)
    parser.add_argument("--reduction", type=str)
    parser.add_argument("--res_depth", type=int, default=3)
    parser.add_argument("--widen_factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    config = build_config(anisotropic=args.anisotropic)

    for _ in range(args.num_experiments):
        train(config)
