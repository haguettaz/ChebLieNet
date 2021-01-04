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

NX1, NX2 = (28, 28)

IN_CHANNELS = 1
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20
POOLING_SIZE = 2

EPOCHS = 20
OPTIMIZER = "adam"

def build_config():
    return {
        "batch_size": {"value": },
        "eps": {"value": },
        "K": {"value": },
        "knn": {"value": },
        "learning_rate": {"value": },
        "nx3": {"value": },
        "pooling": {"value": },
        "weight_sigma": {"value": },
        "weight_decay": {"value": },
        "weight_kernel": {"value": },
        "xi": {"value": },
    }



def get_model(nx3, knn, eps, xi, weight_sigma, weight_kernel, K, pooling):
    # Different graphs are for successive pooling layers
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
            grid_size=(NX1 // POOLING_SIZE, NX2 // POOLING_SIZE),
            nx3=nx3,
            weight_kernel=weight_kernel,
            weight_sigma=weight_sigma,
            knn=knn,  # adapt the number of neighbors to the size of the graph
            sigmas=(xi / eps, xi, 1.0),  # adapt the metric kernel to the size of the graph
            weight_comp_device=DEVICE,
        ),
        HyperCubeGraph(
            grid_size=(NX1 // POOLING_SIZE // POOLING_SIZE, NX2 // POOLING_SIZE // POOLING_SIZE),
            nx3=nx3,
            weight_kernel=weight_kernel,
            weight_sigma=weight_sigma,
            knn=knn,  # adapt the number of neighbors to the size of the graph
            sigmas=(xi / eps, xi, 1.0),  # adapt the metric kernel to the size of the graph
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
        config = wandb.config

        train_loader, _ = get_train_val_data_loaders(
            DATASET_NAME, batch_size=config.batch_size, val_ratio=0., data_path=DATA_PATH
        )

        classic_test_loader, rotated_test_loader, flipped_test_loader = get_test_equivariance_data_loader(
            DATASET_NAME, batch_size=config.batch_size, data_path=DATA_PATH)

        model = get_model(
            config.nx3,
            config.knn,
            config.eps,
            config.xi,
            config.weight_sigma,
            config.weight_kernel,
            config.K,
            config.pooling,
        )

        optimizer = get_optimizer(model, OPTIMIZER, config.learning_rate, config.weight_decay)

        loss_fn = nll_loss
        classic_metrics = {"classic_test_accuracy": Accuracy(), "classic_test_loss": Loss(loss_fn)}
        rotated_metrics = {"rotated_test_accuracy": Accuracy(), "rotated_test_loss": Loss(loss_fn)}
        flipped_metrics = {"flipped_test_accuracy": Accuracy(), "flipped_test_loss": Loss(loss_fn)}

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

        classic_evaluator = create_supervised_evaluator(
            L=config.nx3, model=model, metrics=classic_metrics, device=DEVICE, prepare_batch=prepare_batch
        )
        ProgressBar(persist=False, desc="Evaluation").attach(classic_evaluator)

        rotated_evaluator = create_supervised_evaluator(
            L=config.nx3, model=model, metrics=rotated_metrics, device=DEVICE, prepare_batch=prepare_batch
        )
        ProgressBar(persist=False, desc="Evaluation").attach(rotated_evaluator)

        flipped_evaluator = create_supervised_evaluator(
            L=config.nx3, model=model, metrics=flipped_metrics, device=DEVICE, prepare_batch=prepare_batch
        )
        ProgressBar(persist=False, desc="Evaluation").attach(flipped_evaluator)

        # track training with wandb
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, classic_evaluator, classic_test_loader)
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, rotated_evaluator, rotated_test_loader)
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, flipped_evaluator, flipped_test_loader)

        trainer.run(train_loader, max_epochs=EPOCHS)


if __name__ == "__main__":
    config = build_config()
    for _ in range(NUM_RUNS):
        train(config)
