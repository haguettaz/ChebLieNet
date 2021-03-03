import argparse
import math

import torch
import wandb
from gechebnet.data.dataloader import get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, set_sparse_laplacian, wandb_log
from gechebnet.graph.graph import SE2GEGraph
from gechebnet.model.chebnet import WideGEChebNet
from gechebnet.model.reschebnet import WideResGEChebNet
from gechebnet.nn.models.utils import capacity
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR


def build_sweep_config(anisotropic: bool, coupled_sym: bool) -> dict:
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
        "R": {"distribution": "categorical", "values": [2, 3, 4, 5, 6]},
        "knn": {"distribution": "categorical", "values": [8, 16, 32]},
    }

    if anisotropic:
        if coupled_sym:
            parameters["xi"] = {"distribution": "constant", "value": 50.0}
        else:
            parameters["xi"] = {"distribution": "constant", "value": 1e-4}

        parameters["ntheta"] = {"distribution": "int_uniform", "min": 3, "max": 9}
        parameters["eps"] = {"distribution": "constant", "value": 0.1}

    else:
        parameters["xi"] = {"distribution": "constant", "value": 1.0}
        parameters["ntheta"] = {"distribution": "constant", "value": 1}
        parameters["eps"] = {"distribution": "constant", "value": 1.0}

    sweep_config["parameters"] = parameters

    return sweep_config


def train(config=None):
    """
    Trains a model on CIFAR-10 and evaluates its performance on CIFAR-10, Flip-CIFAR-10 and 90Rot-CIFAR-10.

    Args:
        config (dict, optional): configuration dictionnary. Defaults to None.
    """

    # Initialize a new wandb run
    with wandb.init(config=config):

        config = wandb.config
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Loads graph manifold and set normalized laplacian
        graph = SE2GEGraph(
            nx=32,
            ny=32,
            ntheta=config.ntheta,
            knn=config.knn,
            sigmas=(config.xi / config.eps, config.xi, 1.0),
            weight_kernel=lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac),
        )
        graph.set_laplacian(norm=True, device=device)
        wandb.log({"num_nodes": graph.num_nodes, "num_edges": graph.num_edges})

        # Loads group equivariant Chebnet and optimizer
        if args.resnet:
            model = WideResGEChebNet(
                in_channels=3,
                out_channels=10,
                R=config.R,
                graph=graph,
                depth=args.depth,
                widen_factor=args.widen_factor,
            ).to(device)

        else:
            model = WideGEChebNet(
                in_channels=3,
                out_channels=10,
                R=config.R,
                graph=graph,
                depth=args.depth,
                widen_factor=args.widen_factor,
            ).to(device)
        wandb.log({"capacity": capacity(model)})

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
        train_loader, val_loader = get_train_val_dataloaders(
            "cifar10",
            batch_size=args.batch_size,
            val_ratio=0.3,
            data_path=args.data_path,
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
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        if args.sparse:
            trainer.add_event_handler(
                Events.EPOCH_STARTED,
                set_sparse_laplacian,
                graph,
                args.sparse_on,
                args.sparse_rate,
            )

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
        if args.sparse:
            trainer.add_event_handler(
                Events.EPOCH_STARTED,
                set_sparse_laplacian,
                graph,
                args.sparse_on,
                args.sparse_rate,
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

    sweep_config = build_sweep_config(
        anisotropic=args.anisotropic,
        coupled_sym=args.coupled_sym,
    )

    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=args.num_experiments)
