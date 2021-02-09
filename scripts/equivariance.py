import argparse
import math

import torch
import wandb
from gechebnet.data.dataloader import get_test_equivariance_dataloaders, get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, set_sparse_laplacian, wandb_log
from gechebnet.graph.graph import SE2GEGraph
from gechebnet.model.chebnet import WideGEChebNet
from gechebnet.model.reschebnet import WideResGEChebNet
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR


def build_config(anisotropic: bool, coupled_sym: bool) -> dict:
    """
    [summary]

    Args:
        anisotropic (bool): [description]
        coupled_sym (bool): [description]

    Returns:
        dict: [description]
    """

    config = {
        "K": 6,
        "eps": 0.1 if anisotropic else 1.0,
        "knn": 32 if anisotropic else 16,
        "nsym": 9 if anisotropic else 1,
        "xi": 0.05 if coupled_sym else 1e-4,
    }

    return config


def train(config=None):

    # Initialize a new wandb run
    with wandb.init(config=config, project="gechebnet"):

        config = wandb.config
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Loads graph manifold and set normalized laplacian
        if args.lie_group == "se2":
            graph = SE2GEGraph(
                nx=28 if args.dataset == "mnist" else 96,
                ny=28 if args.dataset == "mnist" else 96,
                ntheta=config.nsym,
                knn=config.knn,
                sigmas=(config.xi / config.eps, config.xi, 1.0),
                weight_kernel=lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac),
            )
        elif args.lie_group == "so3":
            ...

        wandb.log({"num_nodes": graph.num_nodes, "num_edges": graph.num_edges})

        graph.set_laplacian(norm=True)

        # Loads group equivariant Chebnet and optimizer
        if args.resnet:
            model = WideResGEChebNet(
                in_channels=1 if args.dataset == "mnist" else 3,
                out_channels=10,
                K=config.K,
                graph=graph,
                depth=args.depth,
                widen_factor=args.widen_factor,
            ).to(device)

        else:
            model = WideGEChebNet(
                in_channels=1 if args.dataset == "mnist" else 3,
                out_channels=10,
                K=config.K,
                graph=graph,
                depth=args.depth,
                widen_factor=args.widen_factor,
            ).to(device)

        wandb.log({"capacity": model.capacity})

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
            args.dataset,
            batch_size=args.batch_size,
            val_ratio=0.0,
            data_path=args.data_path,
        )

        (
            classic_test_loader,
            rotated_test_loader,
            flipped_test_loader,
        ) = get_test_equivariance_dataloaders(
            args.dataset, batch_size=args.batch_size, data_path=args.data_path
        )

        # Loads engines and adds handlers
        trainer = create_supervised_trainer(
            graph=graph,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        if args.sparse:
            trainer.add_event_handler(
                Events.EPOCH_STARTED,
                set_sparse_laplacian,
                graph,
                args.sparse_on,
                args.sparse_rate,
            )

        classic_metrics = {"classic_test_accuracy": Accuracy(), "classic_test_loss": Loss(nll_loss)}
        rotated_metrics = {"rotated_test_accuracy": Accuracy(), "rotated_test_loss": Loss(nll_loss)}
        flipped_metrics = {"flipped_test_accuracy": Accuracy(), "flipped_test_loss": Loss(nll_loss)}

        classic_evaluator = create_supervised_evaluator(
            graph=graph,
            model=model,
            metrics=classic_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(classic_evaluator)

        rotated_evaluator = create_supervised_evaluator(
            graph=graph,
            model=model,
            metrics=rotated_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(rotated_evaluator)

        flipped_evaluator = create_supervised_evaluator(
            graph=graph,
            model=model,
            metrics=flipped_metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(flipped_evaluator)

        _ = trainer.add_event_handler(
            Events.EPOCH_COMPLETED, wandb_log, classic_evaluator, classic_test_loader
        )
        _ = trainer.add_event_handler(
            Events.EPOCH_COMPLETED, wandb_log, rotated_evaluator, rotated_test_loader
        )
        _ = trainer.add_event_handler(
            Events.EPOCH_COMPLETED, wandb_log, flipped_evaluator, flipped_test_loader
        )

        # Launchs training
        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--data_path", type=str)
    parser.add_argument("-N", "--num_experiments", type=int)
    parser.add_argument("-E", "--max_epochs", type=int)
    parser.add_argument("-D", "--dataset", type=str, choices=["mnist", "stl10"])
    parser.add_argument("-G", "--lie_group", type=str, choices=["se2", "so3"])
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--anisotropic", action="store_true", default=False)
    parser.add_argument("--coupled_sym", action="store_true", default=False)
    parser.add_argument("--resnet", action="store_true", default=False)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--sparse", action="store_true", default=False)
    parser.add_argument("--sparse_rate", type=float, default=0.8)
    parser.add_argument("--sparse_on", type=str, default="edges", choices=["edges", "nodes"])
    parser.add_argument("--optim", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", default=False)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_steps", type=int, nargs="+", default=[100, 150])
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    config = build_config(
        anisotropic=args.anisotropic,
        coupled_sym=args.coupled_sym,
    )

    for _ in range(args.num_experiments):
        train(config)
