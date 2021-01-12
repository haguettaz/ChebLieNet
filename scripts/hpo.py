import math
import os

import torch
import wandb
from gechebnet.data.dataloader import get_train_val_dataloaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch, wandb_log
from gechebnet.graph.graph import SE2GEGraph
from gechebnet.model.chebnet import GEChebNet
from gechebnet.model.optimizer import get_optimizer
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda")

DATASET_NAME = "mnist"  # "stl10"
VAL_RATIO = 0.1
NX1, NX2 = (28, 28)

IN_CHANNELS = 1
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20

EPOCHS = 20
OPTIMIZER = "adam"

NUM_EXPERIMENTS = 100


def build_sweep_config():
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "validation_accuracy", "goal": "maximize"},
    }

    sweep_config["parameters"] = {
        "batch_size": {"distribution": "q_log_uniform", "min": math.log(8), "max": math.log(256)},
        "eps": {"distribution": "log_uniform", "min": math.log(0.1), "max": math.log(1.0)},
        "K": {"distribution": "q_log_uniform", "min": math.log(2), "max": math.log(32)},
        "kappa": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "knn": {"distribution": "categorical", "values": [2, 4, 8, 16, 32, 64]},
        "learning_rate": {
            "distribution": "log_uniform",
            "min": math.log(1e-5),
            "max": math.log(0.1),
        },
        "nx3": {"distribution": "int_uniform", "min": 3, "max": 9},
        "weight_sigma": {"distribution": "uniform", "min": 0.25, "max": 8.0},
        "weight_decay": {
            "distribution": "log_uniform",
            "min": math.log(1e-6),
            "max": math.log(1e-3),
        },
        "weight_kernel": {
            "distribution": "categorical",
            "values": ["cauchy", "gaussian", "laplacian"],
        },
        "xi": {"distribution": "log_uniform", "min": math.log(1e-2), "max": math.log(1.0)},
    }

    return sweep_config


def get_model(nx3, knn, eps, xi, weight_kernel, weight_sigma, kappa, K):
    if weight_kernel == "gaussian":
        kernel = lambda sqdistc: torch.exp(-sqdistc / weight_sigma ** 2)
    elif weight_kernel == "laplacian":
        kernel = lambda sqdistc: torch.exp(-torch.sqrt(sqdistc) / weight_sigma)
    elif weight_kernel == "cauchy":
        kernel = lambda sqdistc: 1 / (1 + sqdistc / weight_sigma ** 2)

    # Different graphs are for successive pooling layers
    graph = SE2GEGraph(
        grid_size=(NX1, NX2),
        nx3=nx3,
        kappa=kappa,
        knn=knn,
        sigmas=(xi / eps, xi, 1.0),
        weight_kernel=kernel,
    )
    if graph.num_nodes > graph.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")
    wandb.log({f"num_nodes": graph.num_nodes, f"num_edges": graph.num_edges})

    model = GEChebNet(
        graph,
        K,
        IN_CHANNELS,
        OUT_CHANNELS,
        HIDDEN_CHANNELS,
        laplacian_device=DEVICE,
    )
    wandb.log({"capacity": model.capacity})

    return model.to(DEVICE)


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config

        # Model and optimizer
        model = get_model(
            config.nx3,
            config.knn,
            config.eps,
            config.xi,
            config.weight_kernel,
            config.weight_sigma,
            config.kappa,
            config.K,
        )

        optimizer = get_optimizer(model, OPTIMIZER, config.learning_rate, config.weight_decay)

        # Trainer and evaluator(s) engines
        trainer = create_supervised_trainer(
            L=config.nx3,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        metrics = {"validation_accuracy": Accuracy(), "validation_loss": Loss(nll_loss)}

        evaluator = create_supervised_evaluator(
            L=config.nx3, model=model, metrics=metrics, device=DEVICE, prepare_batch=prepare_batch
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
