import math
import os

import torch
import torch.nn.functional as F
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

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "MNIST"
VAL_RATIO = 0.3
NX1, NX2 = (28, 28)

IN_CHANNELS = 1
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20

EPOCHS = 20
OPTIMIZER = "adam"


def build_sweep_config():
    sweep_config = {"method": "bayes", "metric": {"name": "val_mnist_acc", "goal": "maximize"}}

    parameters_dict = {
        "batch_size": {"distribution": "q_log_uniform", "min": math.log(16), "max": math.log(256)},
        "eps": {"distribution": "log_uniform", "min": math.log(1e-2), "max": math.log(1.0)},
        "K": {"distribution": "int_uniform", "min": 5, "max": 20},
        "knn": {"distribution": "q_log_uniform", "min": math.log(8), "max": math.log(64)},
        "learning_rate": {"distribution": "log_uniform", "min": math.log(1e-5), "max": math.log(1e-2)},
        "nx3": {"distribution": "int_uniform", "min": 2, "max": 12},
        "pooling": {"values": ["max", "avg"]},
        "weight_sigma": {"distribution": "log_uniform", "min": math.log(0.2), "max": math.log(5.0)},
        "weight_decay": {"distribution": "log_uniform", "min": math.log(1e-6), "max": math.log(1e-3)},
        "weight_kernel": {"values": ["cauchy", "gaussian", "laplacian"]},
        "xi": {"distribution": "log_uniform", "min": math.log(1e-3), "max": math.log(1.0)},
    }
    sweep_config["parameters"] = parameters_dict

    return sweep_config


def get_model(nx3, knn, eps, xi, weight_sigma, weight_kernel, K, pooling):
    graphs = [
        HyperCubeGraph(
            grid_size=(NX1, NX2),
            nx3=nx3,
            weight_kernel=weight_kernel,
            weight_sigma=weight_sigma,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_comp_device=DEVICE,
        ),
        HyperCubeGraph(
            grid_size=(NX1 // 2, NX2 // 2),
            spatial_step=2.0,
            nx3=nx3,
            weight_kernel=weight_kernel,
            weight_sigma=weight_sigma,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_comp_device=DEVICE,
        ),
        HyperCubeGraph(
            grid_size=(NX1 // 2 // 2, NX2 // 2 // 2),
            spatial_step=4.0,
            nx3=nx3,
            weight_kernel=weight_kernel,
            weight_sigma=weight_sigma,
            knn=knn,
            sigmas=(xi / eps, xi, 1.0),
            weight_comp_device=DEVICE,
        ),
    ]

    for idx, graph in enumerate(graphs):
        wandb.log({f"graph_{idx}_nodes": graph.num_nodes, f"graph_{idx}_edges": graph.num_edges})

    model = ChebNet(graphs, K, IN_CHANNELS, OUT_CHANNELS, HIDDEN_CHANNELS, laplacian_device=DEVICE, pooling=pooling)

    wandb.log({"capacity": model.capacity})

    return model.to(DEVICE)


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below, this config will be set by Sweep Controller
        config = wandb.config

        train_loader, val_loader = get_train_val_data_loaders(DATASET, batch_size=config.batch_size, val_ratio=VAL_RATIO, data_path=DATA_PATH)

        model = get_model(config.nx3, config.knn, config.eps, config.xi, config.weight_sigma, config.weight_kernel, config.K, config.pooling)

        optimizer = get_optimizer(model, OPTIMIZER, config.learning_rate, config.weight_decay)

        loss_fn = F.nll_loss
        metrics = {"val_mnist_acc": Accuracy(), "val_mnist_loss": Loss(loss_fn)}

        # create ignite's engines
        trainer = create_supervised_trainer(
            nx=(NX1, NX2, config.nx3), model=model, optimizer=optimizer, loss_fn=F.nll_loss, device=DEVICE, prepare_batch=prepare_batch
        )
        # ProgressBar(persist=False, desc="Training").attach(trainer)

        evaluator = create_supervised_evaluator(nx=(NX1, NX2, config.nx3), model=model, metrics=metrics, device=DEVICE, prepare_batch=prepare_batch)
        # ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        # track training with wandb
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

        # save best model
        trainer.run(train_loader, max_epochs=EPOCHS)


if __name__ == "__main__":
    sweep_config = build_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=20)
