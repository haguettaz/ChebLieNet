import argparse
import os

import torch
import wandb
from gechebnet.datas.dataloaders import get_equiv_test_loaders, get_train_val_loaders
from gechebnet.engines.engines import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engines.utils import prepare_batch, sample_edges, sample_nodes, wandb_log
from gechebnet.graphs.graphs import RandomSubGraph, SE2GEGraph
from gechebnet.liegroups.se2 import se2_uniform_sampling
from gechebnet.nn.models.chebnets import WideResGEChebNet
from gechebnet.nn.models.convnets import WideResConvNet
from gechebnet.nn.models.utils import capacity
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss
from torch.optim import Adam


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
        "kernel_size": 4,
        "eps": 0.1 if anisotropic else 1.0,
        "K": 8,
        "ntheta": 6 if anisotropic else 1,
        "xi": 1.0 if not anisotropic else 2.048 / (28 ** 2) if coupled_sym else 1e6,
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
        wandb.log({"dataset": "mnist"})
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Load model and optimizer
        if args.cnn:
            model = WideResConvNet(1, 10, config.kernel_size, args.depth, args.widen_factor, args.pool).to(device)

        else:

            uniform_sampling = se2_uniform_sampling(28, 28, config.ntheta)
            graph = SE2GEGraph(
                uniform_sampling,
                K=config.K,
                sigmas=(1.0, config.eps, config.xi),
                path_to_graph=args.path_to_graph,
            )
            sub_graph = RandomSubGraph(graph)

            # Loads group equivariant Chebnet
            model = WideResGEChebNet(
                in_channels=1,
                out_channels=10,
                kernel_size=config.kernel_size,
                pool=None,
                graph_lvl0=sub_graph,
                graph_lvl1=None,
                graph_lvl2=None,
                depth=args.depth,
                widen_factor=args.widen_factor,
            ).to(device)

        wandb.log({"capacity": capacity(model)})

        optimizer = Adam(model.parameters(), lr=args.lr)

        # Load dataloaders
        train_loader, _ = get_train_val_loaders(
            "mnist",
            batch_size=args.batch_size,
            val_ratio=0.0,
            path_to_data=args.path_to_data,
        )

        (
            classic_test_loader,
            rotated_test_loader,
            flipped_test_loader,
        ) = get_equiv_test_loaders("mnist", batch_size=args.batch_size, path_to_data=args.path_to_data)

        # Load engines
        trainer = create_supervised_trainer(
            graph=sub_graph if not args.cnn else None,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        classic_metrics = {"classic_test_accuracy": Accuracy(), "classic_test_loss": Loss(nll_loss)}
        rotated_metrics = {"rotated_test_accuracy": Accuracy(), "rotated_test_loss": Loss(nll_loss)}
        flipped_metrics = {"flipped_test_accuracy": Accuracy(), "flipped_test_loss": Loss(nll_loss)}

        classic_evaluator = create_supervised_evaluator(
            graph=sub_graph if not args.cnn else None,
            model=model,
            metrics=classic_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(classic_evaluator)

        rotated_evaluator = create_supervised_evaluator(
            graph=sub_graph if not args.cnn else None,
            model=model,
            metrics=rotated_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(rotated_evaluator)

        flipped_evaluator = create_supervised_evaluator(
            graph=sub_graph if not args.cnn else None,
            model=model,
            metrics=flipped_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(flipped_evaluator)

        if args.sample_edges or args.sample_nodes:
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                sub_graph.reinit,
            )

            # consider all nodes and edges for the evaluation
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
                sub_graph.reinit,
            )

            if args.sample_nodes:
                trainer.add_event_handler(
                    Events.ITERATION_STARTED,
                    sub_graph.node_sampling,
                    args.nodes_rate,
                )

            if args.sample_edges:
                trainer.add_event_handler(
                    Events.ITERATION_STARTED,
                    sub_graph.edge_sampling,
                    args.edges_rate,
                )

        if args.save_models:
            gst = lambda *_: trainer.state.epoch
            handler = Checkpoint(
                {"model": model}, DiskSaver(args.path_to_model, create_dir=True), n_saved=2, global_step_transform=gst
            )
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, classic_evaluator, classic_test_loader)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, rotated_evaluator, rotated_test_loader)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, flipped_evaluator, flipped_test_loader)

        # Launchs training
        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_graph", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--num_experiments", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cnn", action="store_true", default=False)
    parser.add_argument("--anisotropic", action="store_true", default=False)
    parser.add_argument("--coupled_sym", action="store_true", default=False)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--sample_edges", action="store_true", default=False)
    parser.add_argument("--edges_rate", type=float, default=1.0)  # rate of edges to sample
    parser.add_argument("--sample_nodes", action="store_true", default=False)
    parser.add_argument("--nodes_rate", type=float, default=1.0)  # rate of nodes to sample
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--save_models", action="store_true", default=False)
    parser.add_argument("--path_to_model", type=str)
    args = parser.parse_args()

    config = build_config(anisotropic=args.anisotropic, coupled_sym=args.coupled_sym, cnn=args.cnn)

    for _ in range(args.num_experiments):
        train(config)
