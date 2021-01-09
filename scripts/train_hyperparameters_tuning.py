import math
import os
import random

import pykeops
import torch
import wandb
from gechebnet.data.dataloader import get_train_val_data_loaders
from gechebnet.engine.engine import create_supervised_evaluator, create_supervised_trainer
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
DEVICE = torch.device("cuda")

DATASET_NAME = "MNIST"  # STL10
VAL_RATIO = 0.2
NX1, NX2 = (28, 28)

IN_CHANNELS = 1
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20
POOLING_SIZE = 2

EPOCHS = 20
OPTIMIZER = "adam"

NUM_ITER = 10


def build_sweep_config():
    sweep_config = {"method": "bayes", "metric": {"name": "validation_accuracy", "goal": "maximize"}}

    sweep_config["parameters"] = {
        "batch_size": {"distribution": "q_log_uniform", "min": math.log(8), "max": math.log(256)},
        "eps": {"distribution": "log_uniform", "min": math.log(0.1), "max": math.log(1.0)},
        "K": {"distribution": "q_log_uniform", "min": math.log(2), "max": math.log(64)},
        "knn": {"distribution": "categorical", "values": [2, 4, 8, 16, 32]},
        "learning_rate": {"distribution": "log_uniform", "min": math.log(1e-5), "max": math.log(0.1)},
        "nx3": {"distribution": "int_uniform", "min": 3, "max": 9},
        "pooling": {"distribution": "categorical", "values": ["max", "avg"]},
        "weight_sigma": {"distribution": "uniform", "min": 0.25, "max": 8.0},
        "weight_decay": {"distribution": "log_uniform", "min": math.log(1e-6), "max": math.log(1e-3)},
        "weight_kernel": {"distribution": "categorical", "values": ["cauchy", "gaussian", "laplacian"]},
        "xi": {"distribution": "log_uniform", "min": math.log(1e-2), "max": math.log(1.0)},
    }

    return sweep_config


def get_model(nx3, knn, eps, xi, weight_sigma, weight_kernel, K, pooling):

    print("nx3", nx3, type(nx3))
    print("knn", knn, type(knn))
    print("eps", eps, type(eps))
    print("xi", xi, type(xi))
    print("weight_sigma", weight_sigma, type(weight_sigma))
    print("weight_kernel", weight_kernel, type(weight_kernel))
    print("K", K, type(K))
    print("pooling", pooling, type(pooling))

    print("NX1, NX2", NX1, NX2)
    print("NX1 // POOLING_SIZE, NX2 // POOLING_SIZE", NX1 // POOLING_SIZE, NX2 // POOLING_SIZE)
    print(
        "NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE// POOLING_SIZE",
        NX1 // POOLING_SIZE // POOLING_SIZE,
        NX2 // POOLING_SIZE // POOLING_SIZE,
    )

    # Different graphs are for successive pooling layers

    graph_1 = HyperCubeGraph(
        grid_size=(NX1, NX2),
        nx3=nx3,
        weight_kernel=weight_kernel,
        weight_sigma=weight_sigma,
        knn=int(knn * POOLING_SIZE ** 4),
        sigmas=(xi / eps, xi, 1.0),
        weight_comp_device=DEVICE,
    )
    if graph_1.num_nodes > graph_1.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")
    # wandb.log({f"graph_1_nodes": graph_1.num_nodes, f"graph_1_edges": graph_1.num_edges})

    graph_2 = HyperCubeGraph(
        grid_size=(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
        nx3=nx3,
        weight_kernel=weight_kernel,
        weight_sigma=weight_sigma,
        knn=int(knn * POOLING_SIZE ** 2),
        sigmas=(xi / eps, xi, 1.0),
        weight_comp_device=DEVICE,
    )
    if graph_2.num_nodes > graph_2.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")
    # wandb.log({f"graph_2_nodes": graph_2.num_nodes, f"graph_2_edges": graph_2.num_edges})

    graph_3 = HyperCubeGraph(
        grid_size=(NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
        nx3=nx3,
        weight_kernel=weight_kernel,
        weight_sigma=weight_sigma,
        knn=int(knn * POOLING_SIZE ** 4),
        sigmas=(xi / eps, xi, 1.0),
        weight_comp_device=DEVICE,
    )
    if graph_3.num_nodes > graph_3.num_edges:
        raise ValueError(f"An error occured during the computation of the graph")
    # wandb.log({f"graph_3_nodes": graph_3.num_nodes, f"graph_3_edges": graph_3.num_edges})

    model = ChebNet(
        (graph_1, graph_2, graph_3),
        K,
        IN_CHANNELS,
        OUT_CHANNELS,
        HIDDEN_CHANNELS,
        laplacian_device=DEVICE,
        pooling=pooling,
    )

    # wandb.log({"capacity": model.capacity})

    return model.to(DEVICE)


# def train(config=None):
#     # Initialize a new wandb run
#     with wandb.init(config=config):
#         config = wandb.config

#         # Model and optimizer
#         model = get_model(
#             config.nx3,
#             config.knn,
#             config.eps,
#             config.xi,
#             config.weight_sigma,
#             config.weight_kernel,
#             config.K,
#             config.pooling,
#         )

#         optimizer = get_optimizer(model, OPTIMIZER, config.learning_rate, config.weight_decay)
#         loss_fn = nll_loss

#         # Trainer and evaluator(s) engines
#         trainer = create_supervised_trainer(
#             L=config.nx3,
#             model=model,
#             optimizer=optimizer,
#             loss_fn=loss_fn,
#             device=DEVICE,
#             prepare_batch=prepare_batch,
#         )
#         ProgressBar(persist=False, desc="Training").attach(trainer)

#         metrics = {"validation_accuracy": Accuracy(), "validation_loss": Loss(loss_fn)}

#         evaluator = create_supervised_evaluator(
#             L=config.nx3, model=model, metrics=metrics, device=DEVICE, prepare_batch=prepare_batch
#         )
#         ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

#         train_loader, val_loader = get_train_val_data_loaders(
#             DATASET_NAME, batch_size=config.batch_size, val_ratio=VAL_RATIO, data_path=DATA_PATH
#         )

#         # Performance tracking with wandb
#         trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

#         trainer.run(train_loader, max_epochs=EPOCHS)


def train():

    # graph
    nx3 = random.randint(3, 12)
    eps = math.exp(random.uniform(math.log(0.1), math.log(1.0)))
    xi = math.exp(random.uniform(math.log(1e-2), math.log(1.0)))
    knn = random.choice([2, 4, 8, 16, 32])
    weight_kernel = random.choice(["cauchy", "gaussian", "laplacian"])
    weight_sigma = math.exp(random.uniform(math.log(0.25), math.log(10)))

    # network
    K = random.choice([2, 4, 8, 16, 32, 64])
    pooling = random.choice(["max", "avg"])

    # training
    batch_size = random.choice([8, 16, 32, 64, 128, 256])
    learning_rate = math.exp(random.uniform(math.log(1e-5), math.log(0.1)))
    weight_decay = math.exp(random.uniform(math.log(1e-7), math.log(1e-2)))

    # Model and optimizer
    model = get_model(
        nx3,
        knn,
        eps,
        xi,
        weight_sigma,
        weight_kernel,
        K,
        pooling,
    )

    optimizer = get_optimizer(model, OPTIMIZER, learning_rate, weight_decay)
    loss_fn = nll_loss

    # Trainer and evaluator(s) engines
    trainer = create_supervised_trainer(
        L=nx3,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        prepare_batch=prepare_batch,
    )
    ProgressBar(persist=False, desc="Training").attach(trainer)

    metrics = {"validation_accuracy": Accuracy(), "validation_loss": Loss(loss_fn)}

    evaluator = create_supervised_evaluator(
        L=nx3, model=model, metrics=metrics, device=DEVICE, prepare_batch=prepare_batch
    )
    ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

    train_loader, val_loader = get_train_val_data_loaders(
        DATASET_NAME, batch_size=batch_size, val_ratio=VAL_RATIO, data_path=DATA_PATH
    )

    # Performance tracking with wandb
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

    # trainer.run(train_loader, max_epochs=EPOCHS)


if __name__ == "__main__":
    # sweep_config = build_sweep_config()
    # sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    # wandb.agent(sweep_id, train, count=50)
    train()
