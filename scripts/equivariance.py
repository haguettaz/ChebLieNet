import argparse
import math
import os

import torch
import wandb
from gechebnet.data.dataloader import get_test_equivariance_dataloaders, get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, wandb_log
from gechebnet.graph.utils import get_graph
from gechebnet.model.utils import get_model, get_optimizer
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda")


def build_config(anisotropic: bool, coupled_sym: bool, resnet: bool, dataset: str) -> dict:
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

    if anisotropic and coupled_sym and not resnet and dataset == "mnist":
        return {
            "batch_size": {"value": 64},
            "eps": {"value": 0.1},
            "K": {"value": 12},
            "knn": {"value": 32},
            "learning_rate": {"value": 5e-3},
            "nsym": {"value": 9},
            "pooling": {"value": "max"},
            "weight_decay": {"value": 5e-4},
            "xi": {"value": 5e-2},
        }

    elif anisotropic and not coupled_sym and not resnet and dataset == "mnist":
        return {
            "batch_size": {"value": 64},
            "eps": {"value": 0.1},
            "K": {"value": 12},
            "knn": {"value": 32},
            "learning_rate": {"value": 5e-3},
            "nsym": {"value": 9},
            "pooling": {"value": "max"},
            "weight_decay": {"value": 5e-4},
            "xi": {"value": 1e-4},
        }

    elif not anisotropic and not resnet and dataset == "mnist":
        return {
            "batch_size": {"value": 64},
            "eps": {"value": 1.0},
            "K": {"value": 12},
            "knn": {"value": 16},
            "learning_rate": {"value": 5e-3},
            "nsym": {"value": 1},
            "pooling": {"value": "max"},
            "weight_decay": {"value": 5e-4},
            "xi": {"value": 1.0},
        }


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
        wandb.log({"capacity": model.capacity})

        optimizer = get_optimizer(model, config.learning_rate, config.weight_decay)

        # create ignite's engines
        trainer = create_supervised_trainer(
            graph=graph,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        classic_metrics = {"classic_test_accuracy": Accuracy(), "classic_test_loss": Loss(nll_loss)}
        rotated_metrics = {"rotated_test_accuracy": Accuracy(), "rotated_test_loss": Loss(nll_loss)}
        flipped_metrics = {"flipped_test_accuracy": Accuracy(), "flipped_test_loss": Loss(nll_loss)}

        classic_evaluator = create_supervised_evaluator(
            graph=graph,
            model=model,
            metrics=classic_metrics,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(classic_evaluator)

        rotated_evaluator = create_supervised_evaluator(
            graph=graph,
            model=model,
            metrics=rotated_metrics,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(rotated_evaluator)

        flipped_evaluator = create_supervised_evaluator(
            graph=graph,
            model=model,
            metrics=flipped_metrics,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(flipped_evaluator)

        train_loader, _ = get_train_val_dataloaders(
            args.dataset,
            batch_size=config.batch_size,
            val_ratio=0.0,
            data_path=DATA_PATH,
        )

        (
            classic_test_loader,
            rotated_test_loader,
            flipped_test_loader,
        ) = get_test_equivariance_dataloaders(
            args.dataset, batch_size=config.batch_size, data_path=DATA_PATH
        )

        # track training with wandb
        _ = trainer.add_event_handler(
            Events.EPOCH_COMPLETED, wandb_log, classic_evaluator, classic_test_loader
        )
        _ = trainer.add_event_handler(
            Events.EPOCH_COMPLETED, wandb_log, rotated_evaluator, rotated_test_loader
        )
        _ = trainer.add_event_handler(
            Events.EPOCH_COMPLETED, wandb_log, flipped_evaluator, flipped_test_loader
        )

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

    config = build_config(
        anisotropic=args.anisotropic > 0,
        coupled_sym=args.coupled_sym > 0,
        resnet=args.resnet > 0,
        dataset=args.dataset,
    )

    for _ in range(args.num_experiments):
        train(config)
