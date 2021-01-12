import math
import os
import random

import numpy as np
import torch
import torchvision
import wandb
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
from gechebnet.engine.utils import prepare_batch
from gechebnet.graph.graph import SE2GEGraph
from gechebnet.model.chebnet import GEChebNet
from gechebnet.model.optimizer import get_optimizer
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss

print(torchvision.__version__)
# bug somewhere in :
# - torch.utils.data import DataLoader
# - torch.utils.data.sampler import SubsetRandomSampler
# - torchvision.datasets import MNIST, STL10
# - torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, ToTensor)


NX1, NX2 = (28, 28)
DATASET = "mnist"
VAL_RATIO = 0.2
IN_CHANNELS = 1
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20
POOLING_SIZE = 2

OPTIMIZER = "adam"

DEVICE = torch.device("cuda")


def get_model(nx3, knn, eps, xi, weight_kernel, weight_sigma, K, pooling):

    if weight_kernel == "gaussian":
        kernel = lambda sqdistc: torch.exp(-sqdistc / weight_sigma ** 2)
    elif weight_kernel == "laplacian":
        kernel = lambda sqdistc: torch.exp(-torch.sqrt(sqdistc) / weight_sigma)
    elif weight_kernel == "cauchy":
        kernel = lambda sqdistc: 1 / (1 + sqdistc / weight_sigma ** 2)

    # Different graphs are for successive pooling layers
    graph_1 = SE2GEGraph(
        grid_size=(NX1, NX2),
        nx3=nx3,
        knn=int(knn * POOLING_SIZE ** 4),
        sigmas=(xi / eps, xi, 1.0),
        weight_kernel=kernel,
    )
    if graph_1.num_nodes > graph_1.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")

    graph_2 = SE2GEGraph(
        grid_size=(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
        nx3=nx3,
        knn=int(knn * POOLING_SIZE ** 2),
        sigmas=(xi / eps, xi, 1.0),
        weight_kernel=kernel,
    )
    if graph_2.num_nodes > graph_2.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")

    graph_3 = SE2GEGraph(
        grid_size=(NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
        nx3=nx3,
        knn=int(knn * POOLING_SIZE ** 4),
        sigmas=(xi / eps, xi, 1.0),
        weight_kernel=kernel,
    )
    if graph_3.num_nodes > graph_3.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")

    model = GEChebNet(
        (graph_1, graph_2, graph_3),
        K,
        IN_CHANNELS,
        OUT_CHANNELS,
        HIDDEN_CHANNELS,
        laplacian_device=DEVICE,
        pooling=pooling,
    )

    return model.to(DEVICE)


def build_sweep_config():
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "validation_accuracy", "goal": "maximize"},
    }

    sweep_config["parameters"] = {
        "batch_size": {"distribution": "q_log_uniform", "min": math.log(8), "max": math.log(256)},
        "eps": {"distribution": "log_uniform", "min": math.log(0.1), "max": math.log(1.0)},
        "K": {"distribution": "q_log_uniform", "min": math.log(2), "max": math.log(64)},
        "knn": {"distribution": "categorical", "values": [2, 4, 8]},  # 16, 32
        "learning_rate": {
            "distribution": "log_uniform",
            "min": math.log(1e-5),
            "max": math.log(0.1),
        },
        "nx3": {"distribution": "int_uniform", "min": 3, "max": 12},
        "pooling": {"distribution": "categorical", "values": ["max", "avg"]},
        "weight_sigma": {"distribution": "uniform", "min": 0.1, "max": 10},
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
            config.K,
            config.pooling,
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

        metrics = {"validation_accuracy": Accuracy(), "validation_loss": Loss(nll_loss)}

        evaluator = create_supervised_evaluator(
            L=config.nx3, model=model, metrics=metrics, device=DEVICE, prepare_batch=prepare_batch
        )


if __name__ == "__main__":
    sweep_config = build_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=50)
