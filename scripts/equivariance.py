import argparse
import math

import torch
import wandb
from gechebnet.data.dataloader import get_test_equivariance_dataloaders, get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, wandb_log
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss

from .utils import get_graph, get_model, get_optimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    config = {
        "batch_size": 64,
        "K": 6,
        "learning_rate": 5e-3,
        "pooling": "max",
        "weight_decay": 5e-4,
    }

    config["eps"] = 0.1 if anisotropic else 1.0
    config["knn"] = 32 if anisotropic else 16
    config["nsym"] = 9 if anisotropic else 1
    config["xi"] = 0.05 if coupled_sym else 1e-4

    return config


def train(config=None):

    # Initialize a new wandb run
    with wandb.init(config=config):
        print(config)
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

        train_loader, _ = get_train_val_dataloaders(
            args.dataset,
            batch_size=config.batch_size,
            val_ratio=0.0,
            data_path=args.data_path,
        )

        (
            classic_test_loader,
            rotated_test_loader,
            flipped_test_loader,
        ) = get_test_equivariance_dataloaders(
            args.dataset, batch_size=config.batch_size, data_path=args.data_path
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

        # trainer.run(train_loader, max_epochs=args.max_epochs)


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

    config = build_config(
        anisotropic=args.anisotropic > 0,
        coupled_sym=args.coupled_sym > 0,
        resnet=args.resnet > 0,
        dataset=args.dataset,
    )

    for _ in range(args.num_experiments):
        print(config)
        train(config)
