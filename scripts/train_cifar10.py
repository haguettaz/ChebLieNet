import argparse
import math

import torch
import wandb
from gechebnet.data.dataloader import get_test_equivariance_dataloaders, get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import edges_dropout, nodes_sparsification, prepare_batch, wandb_log
from gechebnet.graph.graph import RandomSubGraph, SE2GEGraph
from gechebnet.nn.chebnet import WideGEChebNet, WideResGEChebNet
from gechebnet.nn.cnn import WideCNN, WideResCNN
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR


def build_config(anisotropic: bool, coupled_sym: bool, cnn: bool) -> dict:
    """
    Gets training configuration.

    Args:
        anisotropic (bool): if True, use an anisotropic graph manifold.
        coupled_sym (bool): if True, use coupled symmetric layers.
        cnn (bool): if True, use a convolutional neural network.

    Returns:
        (dict): configuration dictionnary.
    """

    if cnn:
        return {"kernel_size": 3}

    return {
        "R": 4,
        "eps": 0.1 if anisotropic else 1.0,
        "K": 16 if anisotropic else 8,
        "ntheta": 6 if anisotropic else 1,
        "xi": 1.0 if not anisotropic else 50.0 if coupled_sym else 1e-4,
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
        wandb.log({"dataset": "cifar10"})
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        if not args.cnn:

            # Loads graph manifold and set normalized laplacian
            graph_lvl1 = SE2GEGraph(
                nx=32,
                ny=32,
                ntheta=config.ntheta,
                K=config.K,
                sigmas=(config.xi / config.eps, config.xi, 1.0),
                weight_kernel=lambda sqdistc, tc: torch.exp(-sqdistc / 4 * tc),
            )
            sub_graph_lvl1 = RandomSubGraph(graph_lvl1)

            graph_lvl2 = SE2GEGraph(
                nx=16 if args.pool else 32,
                ny=16 if args.pool else 32,
                ntheta=config.ntheta,
                K=config.K,
                sigmas=(config.xi / 4 / config.eps, config.xi / 4, 1.0)
                if args.pool
                else (config.xi / config.eps, config.xi, 1.0),
                weight_kernel=lambda sqdistc, tc: torch.exp(-sqdistc / 4 * tc),
            )
            sub_graph_lvl2 = RandomSubGraph(graph_lvl2)

            graph_lvl3 = SE2GEGraph(
                nx=8 if args.pool else 32,
                ny=8 if args.pool else 32,
                ntheta=config.ntheta,
                K=config.K,
                sigmas=(config.xi / 16 / config.eps, config.xi / 16, 1.0)
                if args.pool
                else (config.xi / config.eps, config.xi, 1.0),
                weight_kernel=lambda sqdistc, tc: torch.exp(-sqdistc / 4 * tc),
            )
            sub_graph_lvl3 = RandomSubGraph(graph_lvl3)

            # Loads group equivariant Chebnet and optimizer
            if args.resnet:
                model = WideResGEChebNet(
                    sub_graph_lvl1,
                    sub_graph_lvl2,
                    sub_graph_lvl3,
                    3,
                    10,
                    config.R,
                    args.depth,
                    args.widen_factor,
                ).to(device)

            else:
                model = WideGEChebNet(
                    sub_graph_lvl1,
                    sub_graph_lvl2,
                    sub_graph_lvl3,
                    3,
                    10,
                    config.R,
                    args.depth,
                    args.widen_factor,
                ).to(device)

        else:
            if args.resnet:
                model = WideResCNN(3, 10, config.kernel_size, args.depth, args.widen_factor, args.pool).to(device)

            else:
                model = WideCNN(3, 10, config.kernel_size, args.depth, args.widen_factor, args.pool).to(device)

        wandb.log({"capacity": model.capacity})

        # Loads optimizer
        if args.optim == "adam":
            optimizer = Adam(model.parameters(), lr=args.lr)
        elif args.optim == "sgd":
            optimizer = SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.decay,
                nesterov=args.nesterov,
            )
        step_scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
        scheduler = LRScheduler(step_scheduler)

        # Loads data loaders
        train_loader, _ = get_train_val_dataloaders(
            "cifar10",
            batch_size=args.batch_size,
            val_ratio=0.0,
            data_path=args.data_path,
        )

        (
            classic_test_loader,
            rotated_test_loader,
            flipped_test_loader,
        ) = get_test_equivariance_dataloaders("cifar10", batch_size=args.batch_size, data_path=args.data_path)

        # Loads engines
        trainer = create_supervised_trainer(
            graph=sub_graph_lvl1 if not args.cnn else None,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

        if args.sample_edges:
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

        if args.sample_nodes:
            trainer.add_event_handler(
                Events.EPOCH_STARTED,
                nodes_sparsification,
                sub_graph_lvl1,
                args.nodes_rate,
            )
            trainer.add_event_handler(
                Events.EPOCH_STARTED,
                nodes_sparsification,
                sub_graph_lvl2,
                args.nodes_rate,
            )
            trainer.add_event_handler(
                Events.EPOCH_STARTED,
                nodes_sparsification,
                sub_graph_lvl3,
                args.nodes_rate,
            )

        classic_metrics = {"classic_test_accuracy": Accuracy(), "classic_test_loss": Loss(nll_loss)}
        rotated_metrics = {"rotated_test_accuracy": Accuracy(), "rotated_test_loss": Loss(nll_loss)}
        flipped_metrics = {"flipped_test_accuracy": Accuracy(), "flipped_test_loss": Loss(nll_loss)}

        classic_evaluator = create_supervised_evaluator(
            graph=sub_graph_lvl1 if not args.cnn else None,
            model=model,
            metrics=classic_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(classic_evaluator)

        rotated_evaluator = create_supervised_evaluator(
            graph=sub_graph_lvl1 if not args.cnn else None,
            model=model,
            metrics=rotated_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(rotated_evaluator)

        flipped_evaluator = create_supervised_evaluator(
            graph=sub_graph_lvl1 if not args.cnn else None,
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
    parser.add_argument("-f", "--data_path", type=str)
    parser.add_argument("-N", "--num_experiments", type=int)
    parser.add_argument("-E", "--max_epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cnn", action="store_true", default=False)
    parser.add_argument("--anisotropic", action="store_true", default=False)
    parser.add_argument("--coupled_sym", action="store_true", default=False)
    parser.add_argument("--resnet", action="store_true", default=False)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--sample_edges", action="store_true", default=False)
    parser.add_argument("--edges_rate", type=float, default=1.0)  # rate of edges or nodes to sample
    parser.add_argument("--sample_nodes", action="store_true", default=False)
    parser.add_argument("--nodes_rate", type=float, default=1.0)  # rate of edges or nodes to sample
    parser.add_argument("--pool", action="store_true", default=False)
    parser.add_argument("--optim", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", default=False)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_steps", type=int, nargs="+", default=[100, 150])
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    config = build_config(anisotropic=args.anisotropic, coupled_sym=args.coupled_sym, cnn=args.cnn)

    for _ in range(args.num_experiments):
        train(config)
