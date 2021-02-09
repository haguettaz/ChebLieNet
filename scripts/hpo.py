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
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss
from torch.optim import Adam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sweep_config(anisotropic: bool, coupled_sym: bool, resnet: bool, dataset: str) -> dict:
    """
    [summary]

    Args:
        anisotropic (bool): [description]
        coupled_sym (bool): [description]
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
            "max": math.log(64) if dataset == "mnist" else math.log(32),
        },
        "K": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 6 if dataset == "mnist" else 10,
        },
        "knn": {"distribution": "categorical", "values": [4, 8, 16, 32]},
        "learning_rate": {
            "distribution": "log_uniform",
            "min": math.log(1e-4),
            "max": math.log(1e-2),
        },
        "weight_decay": {
            "distribution": "log_uniform",
            "min": math.log(1e-6),
            "max": math.log(1e-4),
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


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):

        config = wandb.config

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

        if args.sparsification_rate == 0.0:
            graph.set_laplacian(norm=True)

        wandb.log({f"num_nodes": graph.num_nodes, f"num_edges": graph.num_edges})

        # Loads group equivariant Chebnet and optimizer
        if args.resnet:
            model = WideResGEChebNet(
                in_channels=1 if args.dataset == "mnist" else 3,
                out_channels=10,
                K=config.K,
                graph=graph,
                depth=args.depth,
                widen_factor=args.widen_factor,
            ).to(DEVICE)

        else:
            model = WideGEChebNet(
                in_channels=1 if args.dataset == "mnist" else 3,
                out_channels=10,
                K=config.K,
                graph=graph,
                depth=args.depth,
                widen_factor=args.widen_factor,
            ).to(DEVICE)

        optimizer = Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        wandb.log({"anisotropic": args.anisotropic > 0})
        wandb.log({"coupled_sym": args.coupled_sym > 0})
        wandb.log({"resnet": args.resnet > 0})
        wandb.log({"sparsification_rate": args.sparsification_rate})
        wandb.log({"sparsify_on": args.sparsify_on})
        wandb.log({"capacity": model.capacity})

        # Loads data loaders
        train_loader, val_loader = get_train_val_dataloaders(
            args.dataset,
            batch_size=config.batch_size,
            val_ratio=0.1,
            data_path=args.data_path,
        )

        # Loads engines and adds handlers
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

        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)
        if args.sparsification_rate > 0.0:
            trainer.add_event_handler(
                Events.EPOCH_STARTED,
                set_sparse_laplacian,
                graph,
                args.sparsify_on,
                args.sparsification_rate,
            )

        # Launchs training
        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--data_path", type=str)
    parser.add_argument("-N", "--num_experiments", type=int)
    parser.add_argument("-E", "--max_epochs", type=int)
    parser.add_argument("-D", "--dataset", type=str)
    parser.add_argument("-a", "--anisotropic", type=int, default=0)  # 0: false 1: true
    parser.add_argument("-s", "--coupled_sym", type=int, default=1)  # 0: false 1: true
    parser.add_argument("-r", "--resnet", type=int, default=0)  # 0: false 1: true
    parser.add_argument("-d", "--depth", type=int)
    parser.add_argument("-w", "--widen_factor", type=int)
    parser.add_argument("-g", "--lie_group", type=str)
    parser.add_argument("-k", "--sparsification_rate", type=float, default=0.0)
    parser.add_argument("-o", "--sparsify_on", type=str, default="edges")
    args = parser.parse_args()

    sweep_config = build_sweep_config(
        anisotropic=args.anisotropic > 0,
        coupled_sym=args.coupled_sym > 0,
        resnet=args.resnet > 0,
        dataset=args.dataset,
    )

    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=args.num_experiments)
