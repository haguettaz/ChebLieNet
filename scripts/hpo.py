import argparse
import math

import torch
import wandb
from gechebnet.data.dataloader import get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, sparsify_laplacian, wandb_log
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss

from .utils import get_graph, get_model, get_optimizer

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
            "distribution": "q_log_uniform",
            "min": math.log(2),
            "max": math.log(8) if dataset == "mnist" else math.log(16),
        },
        "knn": {"distribution": "categorical", "values": [4, 8, 16, 32]},
        "learning_rate": {
            "distribution": "log_uniform",
            "min": math.log(1e-4),
            "max": math.log(1e-2),
        },
        "pooling": {"distribution": "categorical", "values": ["avg", "max"]},
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

        graph = get_graph(
            lie_group=args.lie_group,
            dataset=args.dataset,
            nsym=config.nsym,
            knn=config.knn,
            eps=config.eps,
            xi=config.xi,
            device=DEVICE,
        )
        wandb.log({f"num_nodes": graph.num_nodes, f"num_edges": graph.num_edges})

        model = get_model(
            graph=graph,
            in_channels=1 if args.dataset == "mnist" else 3,
            hidden_channels=args.hidden_channels,
            out_channels=10,
            K=config.K,
            pooling=config.pooling,
            resnet=args.resnet > 0,
            device=DEVICE,
        )
        wandb.log({"anisotropic": args.anisotropic > 0})
        wandb.log({"coupled_sym": args.coupled_sym > 0})
        wandb.log({"resnet": args.resnet > 0})
        wandb.log({"sparsification_rate": args.sparsification_rate})
        wandb.log({"sparsify_on": args.sparsify_on})
        wandb.log({"capacity": model.capacity})

        optimizer = get_optimizer(model, config.learning_rate, config.weight_decay)

        train_loader, val_loader = get_train_val_dataloaders(
            args.dataset,
            batch_size=config.batch_size,
            val_ratio=0.1,
            data_path=args.data_path,
        )

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

        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)
        if args.sparsification_rate:
            trainer.add_event_handler(
                Events.EPOCH_STARTED,
                sparsify_laplacian,
                model,
                args.sparsify_on,
                args.sparsification_rate,
            )

        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data_path", type=str)
    parser.add_argument("-n", "--num_experiments", type=int)
    parser.add_argument("-m", "--max_epochs", type=int)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-a", "--anisotropic", type=int, default=0)  # 0: false 1: true
    parser.add_argument("-s", "--coupled_sym", type=int, default=1)  # 0: false 1: true
    parser.add_argument("-r", "--resnet", type=int, default=0)  # 0: false 1: true
    parser.add_argument("-c", "--hidden_channels", nargs="+", type=int, action="append")
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
