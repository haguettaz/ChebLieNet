import argparse
import math
import os

import torch
import wandb
from gechebnet.data.dataloader import get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, wandb_log
from gechebnet.graph.utils import get_graph
from gechebnet.model.utils import get_model, get_optimizer
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss

# DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda")


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
            "max": math.log(16) if dataset == "mnist" else math.log(32),
        },
        "knn": {"distribution": "categorical", "values": [4, 8, 16, 32]},
        "learning_rate": {
            "distribution": "log_uniform",
            "min": math.log(1e-5),
            "max": math.log(0.1),
        },
        "pooling": {"distribution": "categorical", "values": ["avg", "max"]},
        "weight_decay": {
            "distribution": "log_uniform",
            "min": math.log(1e-6),
            "max": math.log(1e-3),
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
        torch.cuda.empty_cache()
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
            resnet=args.model_type[2] == "resnet",
            device=DEVICE,
        )
        wandb.log({"anisotropic": args.model_type[0] == "anisotropic"})
        wandb.log({"coupled_sym": args.model_type[1] == "coupled_sym"})
        wandb.log({"resnet": args.model_type[2] == "resnet"})
        wandb.log({"capacity": model.capacity})

        optimizer = get_optimizer(model, config.learning_rate, config.weight_decay)

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

        train_loader, val_loader = get_train_val_dataloaders(
            args.dataset,
            batch_size=config.batch_size,
            val_ratio=0.1,
            data_path=DATA_PATH,
        )

        # Performance tracking with wandb
        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experiments", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--anisotropic", type=int)  # 0: false 1: true
    parser.add_argument("--coupled_sym", type=int)  # 0: false 1: true
    parser.add_argument("--resnet", type=int)  # 0: false 1: true
    parser.add_argument("--hidden_channels", nargs="+", type=int)
    parser.add_argument("--lie_group", type=str)
    args = parser.parse_args()

    sweep_config = build_sweep_config(
        anisotropic=args.anisotropic > 0,
        coupled_sym=args.coupled_sym > 0,
        resnet=args.resnet > 0,
        dataset=args.dataset,
    )

    # sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    # wandb.agent(sweep_id, train, count=args.num_experiments)
