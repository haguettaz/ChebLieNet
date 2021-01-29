import math
import os

import torch
import wandb
from gechebnet.data.dataloader import get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, wandb_log
from gechebnet.graph.graph import SE2GEGraph, SO3GEGraph
from gechebnet.model.chebnet import GEChebNet_v0, GEChebNet_v1
from gechebnet.model.optimizer import get_optimizer
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda")

DATASET_NAME = "mnist"  # "stl10"
VAL_RATIO = 0.1

IN_CHANNELS = 1 if DATASET_NAME == "mnist" else 3
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20

EPOCHS = 20 if DATASET_NAME == "mnist" else 100
OPTIMIZER = "adam"

NUM_EXPERIMENTS = 50

LIE_GROUP = "se2"  # "so3"
MODEL = "gechebnet"  # "gechebnet" "gechebnet_"


def build_sweep_config():
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "validation_accuracy", "goal": "maximize"},
    }

    if not MODEL in {"gechebnet", "chebnet", "gechebnet_"}:
        raise ValueError(
            f"{MODEL} is not a valid value for MODEL: must be 'gechebnet' or 'chebnet', 'gechebnet_'"
        )

    if MODEL == "gechebnet":
        sweep_config["parameters"] = {
            "batch_size": {
                "distribution": "q_log_uniform",
                "min": math.log(8),
                "max": math.log(256),
            },
            "eps": {"distribution": "constant", "value": 0.1},
            "K": {
                "distribution": "q_log_uniform",
                "min": math.log(2),
                "max": math.log(16) if DATASET_NAME == "mnist" else math.log(32),
            },
            "kappa": {"distribution": "constant", "value": 0.0},
            "knn": {"distribution": "categorical", "values": [4, 8, 16, 32]},
            "learning_rate": {
                "distribution": "log_uniform",
                "min": math.log(1e-5),
                "max": math.log(0.1),
            },
            "nsym": {"distribution": "int_uniform", "min": 3, "max": 12},
            "pooling": {"distribution": "categorical", "values": ["avg", "max"]},
            "weight_decay": {
                "distribution": "log_uniform",
                "min": math.log(1e-6),
                "max": math.log(1e-3),
            },
            "xi": {"distribution": "log_uniform", "min": math.log(1e-2), "max": math.log(1.0)},
        }
    elif MODEL == "gechebnet_":
        sweep_config["parameters"] = {
            "batch_size": {
                "distribution": "q_log_uniform",
                "min": math.log(8),
                "max": math.log(256),
            },
            "eps": {"distribution": "constant", "value": 0.1},
            "K": {
                "distribution": "q_log_uniform",
                "min": math.log(2),
                "max": math.log(16) if DATASET_NAME == "mnist" else math.log(32),
            },
            "kappa": {"distribution": "constant", "value": 0.0},
            "knn": {"distribution": "categorical", "values": [4, 8, 16, 32]},
            "learning_rate": {
                "distribution": "log_uniform",
                "min": math.log(1e-5),
                "max": math.log(0.1),
            },
            "nsym": {"distribution": "int_uniform", "min": 3, "max": 12},
            "pooling": {"distribution": "categorical", "values": ["avg", "max"]},
            "weight_decay": {
                "distribution": "log_uniform",
                "min": math.log(1e-6),
                "max": math.log(1e-3),
            },
            "xi": {"distribution": "constant", "value": 1e-4},  # independent symmetry layers
        }

    elif MODEL == "chebnet":
        sweep_config["parameters"] = {
            "batch_size": {
                "distribution": "q_log_uniform",
                "min": math.log(8),
                "max": math.log(256),
            },
            "eps": {"distribution": "constant", "value": 1.0},
            "K": {
                "distribution": "q_log_uniform",
                "min": math.log(2),
                "max": math.log(16) if DATASET_NAME == "mnist" else math.log(32),
            },
            "kappa": {"distribution": "constant", "value": 0.0},
            "knn": {"distribution": "categorical", "values": [4, 8, 16, 32]},
            "learning_rate": {
                "distribution": "log_uniform",
                "min": math.log(1e-5),
                "max": math.log(0.1),
            },
            "nsym": {"distribution": "constant", "value": 1},
            "pooling": {"distribution": "categorical", "values": ["avg", "max"]},
            "weight_decay": {
                "distribution": "log_uniform",
                "min": math.log(1e-6),
                "max": math.log(1e-3),
            },
            "xi": {"distribution": "constant", "value": 1.0},
        }

    return sweep_config


def get_graph(nsym, knn, eps, xi, kappa):
    if LIE_GROUP == "se2":
        graph = SE2GEGraph(
            nx=28 if DATASET_NAME == "mnist" else 96,
            ny=28 if DATASET_NAME == "mnist" else 96,
            ntheta=nsym,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_kernel=lambda sqdistc, sqsigmac: torch.exp(-sqdistc / sqsigmac),
            kappa=kappa,
            device=DEVICE,
        )

    elif LIE_GROUP == "so3":
        graph = SO3GEGraph(
            nsamples=28 * 28 if DATASET_NAME == "mnist" else 96 * 96,
            nalpha=nsym,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_kernel=lambda sqdistc, sigmac: torch.exp(-sqdistc / sigmac),
            kappa=kappa,
            device=DEVICE,
        )

    if graph.num_nodes > graph.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")
    wandb.log({f"num_nodes": graph.num_nodes, f"num_edges": graph.num_edges})

    return graph


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config

        graph = get_graph(config.nsym, config.knn, config.eps, config.xi, config.kappa)

        if DATASET_NAME == "mnist":
            model = GEChebNet_v0(
                graph=graph,
                K=config.K,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                hidden_channels=HIDDEN_CHANNELS,
                pooling=config.pooling,
                device=DEVICE,
            )
        elif DATASET_NAME == "stl10":
            model = GEChebNet_v1(
                graph=graph,
                K=config.K,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                hidden_channels=HIDDEN_CHANNELS,
                pooling=config.pooling,
                device=DEVICE,
            )
        model = model.to(DEVICE)
        optimizer = get_optimizer(model, OPTIMIZER, config.learning_rate, config.weight_decay)
        wandb.log({"capacity": model.capacity})

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
            graph=graph, model=model, metrics=metrics, device=DEVICE, prepare_batch=prepare_batch
        )
        ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        train_loader, val_loader = get_train_val_dataloaders(
            DATASET_NAME, batch_size=config.batch_size, val_ratio=VAL_RATIO, data_path=DATA_PATH
        )

        # Performance tracking with wandb
        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

        trainer.run(train_loader, max_epochs=EPOCHS)


if __name__ == "__main__":
    sweep_config = build_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=NUM_EXPERIMENTS)
