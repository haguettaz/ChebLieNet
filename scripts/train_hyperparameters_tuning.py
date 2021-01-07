import math
import os

import pykeops
import torch
import wandb
from gechebnet.data.dataloader import get_train_val_data_loaders
from gechebnet.engine.engine import (create_supervised_evaluator,
                                     create_supervised_trainer)
from gechebnet.engine.utils import prepare_batch, wandb_log
from gechebnet.graph.graph import HyperCubeGraph
from gechebnet.model.chebnet import ChebNet
from gechebnet.model.optimizer import get_optimizer
from gechebnet.utils import random_choice
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn import NLLLoss
from torch.nn.functional import nll_loss

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "MNIST"  # STL10
VAL_RATIO = 0.2
NX1, NX2 = (28, 28)

IN_CHANNELS = 1
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20
POOLING_SIZE = 2

EPOCHS = 10
OPTIMIZER = "adam"

MIN_KNN = 2
MULT_KNN = 2


def build_sweep_config():
    sweep_config = {"method": "bayes", "metric": {"name": "validation_accuracy", "goal": "maximize"}}

    parameters_dict = {
        "batch_size": {"distribution": "q_log_uniform", "min": math.log(16), "max": math.log(128)},
        "eps": {"distribution": "log_uniform", "min": math.log(1e-2), "max": math.log(1.0)},
        "K": {"distribution": "int_uniform", "min": 2, "max": 20},
        "exp_knn": {"distribution": "int_uniform", "min": 0, "max": 4},
        "learning_rate": {"distribution": "log_uniform", "min": math.log(1e-5), "max": math.log(1e-2)},
        "nx3": {"distribution": "int_uniform", "min": 4, "max": 9},
        "pooling": {"values": ["max", "avg"]},
        "weight_sigma": {"distribution": "log_uniform", "min": math.log(0.5), "max": math.log(5.0)},
        "weight_decay": {"distribution": "log_uniform", "min": math.log(1e-6), "max": math.log(1e-3)},
        "weight_kernel": {"values": ["cauchy", "gaussian", "laplacian"]},
        "xi": {"distribution": "log_uniform", "min": math.log(1e-2), "max": math.log(1.0)},
    }
    sweep_config["parameters"] = parameters_dict

    return sweep_config


def robust_graph_construction(grid_size, nx3, knn, eps, xi, weight_sigma, weight_kernel):

    graph = HyperCubeGraph(
        grid_size=(NX1, NX2),
        nx3=nx3,
        weight_kernel=weight_kernel,
        weight_sigma=weight_sigma,
        knn=knn,
        sigmas=(xi / eps, xi, 1.0),
        weight_comp_device=DEVICE,
    )

    while graph.num_edges < graph.num_nodes:
        print("construction failed")
        graph = HyperCubeGraph(
            grid_size=(NX1, NX2),
            nx3=nx3,
            weight_kernel=weight_kernel,
            weight_sigma=weight_sigma,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_comp_device=DEVICE,
        )

    return graph


def get_model(nx3, exp_knn, eps, xi, weight_sigma, weight_kernel, K, pooling):
    # Different graphs are for successive pooling layers

    graph_1 = robust_graph_construction(
        (NX1, NX2), nx3, int(MIN_KNN * MULT_KNN ** exp_knn * POOLING_SIZE ** 4), eps, xi, weight_sigma, weight_kernel
    )

    wandb.log({f"graph_1_nodes": graph_1.num_nodes, f"graph_1_edges": graph_1.num_edges})

    graph_2 = robust_graph_construction(
        (NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
        nx3,
        int(MIN_KNN * MULT_KNN ** exp_knn * POOLING_SIZE ** 2),
        eps,
        xi,
        weight_sigma,
        weight_kernel,
    )

    wandb.log({f"graph_2_nodes": graph_2.num_nodes, f"graph_2_edges": graph_2.num_edges})

    graph_3 = robust_graph_construction(
        (NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
        nx3,
        int(MIN_KNN * MULT_KNN ** exp_knn),
        eps,
        xi,
        weight_sigma,
        weight_kernel,
    )

    wandb.log({f"graph_3_nodes": graph_3.num_nodes, f"graph_3_edges": graph_3.num_edges})

    model = ChebNet(
        (graph_1, graph_2, graph_3),
        K,
        IN_CHANNELS,
        OUT_CHANNELS,
        HIDDEN_CHANNELS,
        laplacian_device=DEVICE,
        pooling=pooling,
    )

    wandb.log({"capacity": model.capacity})

    return model.to(DEVICE)


def train(config=None):

    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config

        train_loader, val_loader = get_train_val_data_loaders(
            DATASET_NAME, batch_size=config.batch_size, val_ratio=VAL_RATIO, data_path=DATA_PATH
        )

        model = get_model(
            config.nx3,
            config.exp_knn,
            config.eps,
            config.xi,
            config.weight_sigma,
            config.weight_kernel,
            config.K,
            config.pooling,
        )

        optimizer = get_optimizer(model, OPTIMIZER, config.learning_rate, config.weight_decay)

        loss_fn = nll_loss
        metrics = {"validation_accuracy": Accuracy(), "validation_loss": Loss(loss_fn)}

        # create ignite's engines
        trainer = create_supervised_trainer(
            L=config.nx3,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        evaluator = create_supervised_evaluator(
            L=config.nx3, model=model, metrics=metrics, device=DEVICE, prepare_batch=prepare_batch
        )
        ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        # track training with wandb
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

        trainer.run(train_loader, max_epochs=EPOCHS)


if __name__ == "__main__":
    sweep_config = build_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=50)
