import argparse
import os

import torch
import wandb
from gechebnet.datas.dataloaders import get_test_loader, get_train_val_loaders
from gechebnet.engines.engines import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engines.utils import edges_dropout, prepare_batch, wandb_log
from gechebnet.graphs.graphs import RandomSubGraph, SO3GEGraph
from gechebnet.liegroups.so3 import so3_uniform_sampling
from gechebnet.nn.layers.pools import IcosahedralPool
from gechebnet.nn.layers.unpools import IcosahedralUnpool
from gechebnet.nn.models.chebnets import UChebNet
from gechebnet.nn.models.utils import capacity
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, ConfusionMatrix, Fbeta, Loss
from ignite.metrics.confusion_matrix import cmAccuracy, cmPrecision, cmRecall, mIoU
from torch.nn.functional import nll_loss
from torch.optim import Adam


def build_config(anisotropic: bool) -> dict:
    """
    Gets training configuration.

    Args:
        anisotropic (bool): if True, use an anisotropic graph manifold.

    Returns:
        (dict): configuration dictionnary.
    """

    return {
        "R": 3,
        "eps": 0.1 if anisotropic else 1.0,
        "K": 8,
        "nalpha": 6 if anisotropic else 1,
        "xi_0": 1 if not anisotropic else 2.67500,
        "xi_1": 1 if not anisotropic else 0.76429,
        "xi_2": 1 if not anisotropic else 0.19815,
        "xi_3": 1 if not anisotropic else 0.05,
        "xi_4": 1 if not anisotropic else 0.01253,
        "xi_5": 1 if not anisotropic else 0.00313,
    }


def train(config=None):
    """
    Trains a model on MNIST and evaluates its performance on MNIST, Flip-MNIST and 90Rot-MNIST.

    Args:
        config (dict, optional): configuration dictionnary. Defaults to None.
    """

    # Initialize a new wandb run
    with wandb.init(config=config, project="gechebnet"):

        config = wandb.config
        wandb.log({"dataset": "artc"})
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Load model and optimizer
        uniform_sampling_lvl0 = so3_uniform_sampling(args.path_to_sampling, 0, config.nalpha)
        graph_lvl0 = SO3GEGraph(
            uniform_sampling_lvl0,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_0),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl0 = RandomSubGraph(graph_lvl0)

        uniform_sampling_lvl1 = so3_uniform_sampling(args.path_to_sampling, 1, config.nalpha)
        graph_lvl1 = SO3GEGraph(
            uniform_sampling_lvl1,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_1),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl1 = RandomSubGraph(graph_lvl1)

        uniform_sampling_lvl2 = so3_uniform_sampling(args.path_to_sampling, 2, config.nalpha)
        graph_lvl2 = SO3GEGraph(
            uniform_sampling_lvl2,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_2),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl2 = RandomSubGraph(graph_lvl2)

        uniform_sampling_lvl3 = so3_uniform_sampling(args.path_to_sampling, 3, config.nalpha)
        graph_lvl3 = SO3GEGraph(
            uniform_sampling_lvl3,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_3),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl3 = RandomSubGraph(graph_lvl3)

        uniform_sampling_lvl4 = so3_uniform_sampling(args.path_to_sampling, 4, config.nalpha)
        graph_lvl4 = SO3GEGraph(
            uniform_sampling_lvl4,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_4),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl4 = RandomSubGraph(graph_lvl4)

        uniform_sampling_lvl5 = so3_uniform_sampling(args.path_to_sampling, 5, config.nalpha)
        graph_lvl5 = SO3GEGraph(
            uniform_sampling_lvl5,
            K=config.K,
            sigmas=(1.0, config.eps, config.xi_5),
            path_to_graph=args.path_to_graph,
        )
        sub_graph_lvl5 = RandomSubGraph(graph_lvl5)

        # Loads group equivariant Chebnet
        model = UChebNet(
            16,
            3,
            config.R,
            IcosahedralPool,
            IcosahedralUnpool,
            sub_graph_lvl0,
            sub_graph_lvl1,
            sub_graph_lvl2,
            sub_graph_lvl3,
            sub_graph_lvl4,
            sub_graph_lvl5,
        ).to(device)

        wandb.log({"capacity": capacity(model)})

        optimizer = Adam(model.parameters(), lr=args.lr)

        # Load dataloaders
        train_loader, _ = get_train_val_loaders(
            "artc",
            batch_size=args.batch_size,
            val_ratio=0.0,
            path_to_data=args.path_to_data,
        )

        test_loader = get_test_loader("artc", batch_size=args.batch_size, path_to_data=args.path_to_data)

        # Load engines
        trainer = create_supervised_trainer(
            graph=sub_graph_lvl5,
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
                edges_dropout,
                sub_graph_lvl0,
                args.edges_rate,
            )
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                edges_dropout,
                sub_graph_lvl1,
                args.edges_rate,
            )
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                edges_dropout,
                sub_graph_lvl2,
                args.edges_rate,
            )
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                edges_dropout,
                sub_graph_lvl3,
                args.edges_rate,
            )
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                edges_dropout,
                sub_graph_lvl4,
                args.edges_rate,
            )
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                edges_dropout,
                sub_graph_lvl5,
                args.edges_rate,
            )

        cm = ConfusionMatrix(num_classes=3)
        precision = cmPrecision(cm, average=False)
        recall = cmRecall(cm, average=False)
        metrics = {
            "test_F1": Fbeta(1, precision=precision, recall=recall),
            "test_mIoU": mIoU(cm),
            "test_mIoU_nb": mIoU(cm, ignore_index=0),
            "test_loss": Loss(nll_loss),
        }

        evaluator = create_supervised_evaluator(
            graph=sub_graph_lvl5,
            model=model,
            metrics=metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, test_loader)

        # Launchs training
        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_sampling", type=str)
    parser.add_argument("--path_to_graph", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--num_experiments", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--anisotropic", action="store_true", default=False)
    parser.add_argument("--coupled_sym", action="store_true", default=False)
    parser.add_argument("--sample_edges", action="store_true", default=False)
    parser.add_argument("--edges_rate", type=float, default=1.0)  # rate of edges to sample
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    config = build_config(anisotropic=args.anisotropic)

    for _ in range(args.num_experiments):
        train(config)
