import argparse
import os

import torch
import wandb
from gechebnet.datas.dataloaders import get_equiv_test_loaders, get_train_val_loaders
from gechebnet.engines.engines import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engines.utils import prepare_batch, sample_edges, sample_nodes, wandb_log
from gechebnet.graphs.graphs import RandomSubGraph, SE2GEGraph
from gechebnet.liegroups.se2 import se2_uniform_sampling
from gechebnet.nn.layers.pools import CubicPool
from gechebnet.nn.models.chebnets import WideResGEChebNet
from gechebnet.nn.models.convnets import WideResConvNet
from gechebnet.nn.models.utils import capacity
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

    return {
        "kernel_size": 2,
        "eps": 0.1 if anisotropic else 1.0,
        "K": 8,
        "ntheta": 6 if anisotropic else 1,
        "xi_0": 1 if not anisotropic else 2.048 / (24 ** 2),
        "xi_1": 1 if not anisotropic else 2.048 / (48 ** 2),
        "xi_2": 1 if not anisotropic else 2.048 / (96 ** 2),
    }


def train(config=None):
    """
    Trains a model on STL10 and evaluates its performance on STL10, Flip-STL10 and 90Rot-STL10.

    Args:
        config (dict, optional): configuration dictionnary. Defaults to None.
    """

    # Initialize a new wandb run
    with wandb.init(config=config, project="gechebnet"):

        config = wandb.config
        wandb.log({"dataset": "stl10"})
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Load model and optimizer
        uniform_sampling_lvl0 = se2_uniform_sampling(24, 24, config.ntheta)
        graph_lvl0 = SE2GEGraph(
            uniform_sampling_lvl0,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_0),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl0 = RandomSubGraph(graph_lvl0)

        uniform_sampling_lvl1 = se2_uniform_sampling(48, 48, config.ntheta)
        graph_lvl1 = SE2GEGraph(
            uniform_sampling_lvl1,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_1),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl1 = RandomSubGraph(graph_lvl1)

        uniform_sampling_lvl2 = se2_uniform_sampling(96, 96, config.ntheta)
        graph_lvl2 = SE2GEGraph(
            uniform_sampling_lvl2,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_2),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl2 = RandomSubGraph(graph_lvl2)

        # Loads group equivariant Chebnet
        model = WideResGEChebNet(
            in_channels=3,
            out_channels=10,
            kernel_size=config.kernel_size,
            pool=CubicPool,
            graph_lvl0=sub_graph_lvl0,
            graph_lvl1=sub_graph_lvl1,
            graph_lvl2=sub_graph_lvl2,
            depth=args.depth,
            widen_factor=args.widen_factor,
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

        (classic_test_loader, rotated_test_loader, flipped_test_loader,) = get_equiv_test_loaders(
            "stl10", batch_size=args.batch_size, num_layers=config.ntheta, path_to_data=args.path_to_data
        )

        # Load engines
        trainer = create_supervised_trainer(
            graph=sub_graph_lvl2,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        if args.sample_edges:
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                sample_edges,
                sub_graph_lvl0,
                args.edges_rate,
            )
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                sample_edges,
                sub_graph_lvl1,
                args.edges_rate,
            )
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                sample_edges,
                sub_graph_lvl2,
                args.edges_rate,
            )

        classic_metrics = {"classic_test_accuracy": Accuracy(), "classic_test_loss": Loss(nll_loss)}
        rotated_metrics = {"rotated_test_accuracy": Accuracy(), "rotated_test_loss": Loss(nll_loss)}
        flipped_metrics = {"flipped_test_accuracy": Accuracy(), "flipped_test_loss": Loss(nll_loss)}

        classic_evaluator = create_supervised_evaluator(
            graph=sub_graph_lvl2,
            model=model,
            metrics=classic_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(classic_evaluator)

        rotated_evaluator = create_supervised_evaluator(
            graph=sub_graph_lvl2,
            model=model,
            metrics=rotated_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(rotated_evaluator)

        flipped_evaluator = create_supervised_evaluator(
            graph=sub_graph_lvl2,
            model=model,
            metrics=flipped_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(flipped_evaluator)

        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, classic_evaluator, classic_test_loader)
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, rotated_evaluator, rotated_test_loader)
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, flipped_evaluator, flipped_test_loader)

        # Launchs training
        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_graph", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--num_experiments", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--anisotropic", action="store_true", default=False)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--widen_factor", type=int, default=4)
    parser.add_argument("--sample_edges", action="store_true", default=False)
    parser.add_argument("--edges_rate", type=float, default=1.0)  # rate of edges to sample
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    config = build_config(anisotropic=args.anisotropic, coupled_sym=args.coupled_sym)

    for _ in range(args.num_experiments):
        train(config)
