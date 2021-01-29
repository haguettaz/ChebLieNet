import math
import os

import pykeops
import torch
import wandb
from gechebnet.data.dataloader import (get_test_equivariance_dataloaders,
                                       get_train_val_dataloaders)
from gechebnet.engine.engine import (create_supervised_evaluator,
                                     create_supervised_trainer)
from gechebnet.engine.utils import prepare_batch, wandb_log
from gechebnet.graph.graph import SE2GEGraph
from gechebnet.model.chebnet import GEChebNet_v0, GEChebNet_v1
from gechebnet.model.optimizer import get_optimizer
from gechebnet.utils import random_choice
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from torch.nn.functional import nll_loss

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "mnist"  # stl10

IN_CHANNELS = 1
OUT_CHANNELS = 10
HIDDEN_CHANNELS = 20
POOLING_SIZE = 2

EPOCHS = 20
OPTIMIZER = "adam"

MAX_ITER = 5

LIE_GROUP = "se2"  # "so3"
MODEL = "chebnet"  # "gechebnet" "gechebnet_"

def build_config():
    if not MODEL in {"gechebnet", "chebnet", "gechebnet_"}:
        raise ValueError(
            f"{MODEL} is not a valid value for MODEL: must be 'gechebnet' or 'chebnet', 'gechebnet_'"
        )

    if MODEL == "gechebnet":
        return {
            "batch_size": {"value": },
            "eps": {"value": },
            "K": {"value": },
            "knn": {"value": },
            "learning_rate": {"value": },
            "nsym": {"value": },
            "pooling": {"value": },
            "weight_sigma": {"value": },
            "weight_decay": {"value": },
            "xi": {"value": },
        }
    
    elif MODEL == "gechebnet_":
        return {
            "batch_size": {"value": },
            "eps": {"value": },
            "K": {"value": },
            "knn": {"value": },
            "learning_rate": {"value": },
            "nsym": {"value": },
            "pooling": {"value": },
            "weight_sigma": {"value": },
            "weight_decay": {"value": },
            "xi": {"value": },
        }

    elif MODEL == "chebnet":
        return {
            "batch_size": {"value": },
            "eps": {"value": },
            "K": {"value": },
            "knn": {"value": },
            "learning_rate": {"value": },
            "nsym": {"value": },
            "pooling": {"value": },
            "weight_sigma": {"value": },
            "weight_decay": {"value": },
            "xi": {"value": },
        }

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

        train_loader, _ = get_train_val_dataloaders(
            DATASET_NAME, batch_size=config.batch_size, val_ratio=0., data_path=DATA_PATH
        )

        classic_test_loader, rotated_test_loader, flipped_test_loader = get_test_equivariance_dataloaders(
            DATASET_NAME, batch_size=config.batch_size, data_path=DATA_PATH)

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

        optimizer = get_optimizer(model, OPTIMIZER, config.learning_rate, config.weight_decay)

        loss_fn = nll_loss
        classic_metrics = {"classic_test_accuracy": Accuracy(), "classic_test_loss": Loss(loss_fn)}
        rotated_metrics = {"rotated_test_accuracy": Accuracy(), "rotated_test_loss": Loss(loss_fn)}
        flipped_metrics = {"flipped_test_accuracy": Accuracy(), "flipped_test_loss": Loss(loss_fn)}

        # create ignite's engines
        trainer = create_supervised_trainer(
            graph=graph,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=DEVICE,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        classic_evaluator = create_supervised_evaluator(
            L=config.nsym, model=model, metrics=classic_metrics, device=DEVICE, prepare_batch=prepare_batch
        )
        ProgressBar(persist=False, desc="Evaluation").attach(classic_evaluator)

        rotated_evaluator = create_supervised_evaluator(
            L=config.nsym, model=model, metrics=rotated_metrics, device=DEVICE, prepare_batch=prepare_batch
        )
        ProgressBar(persist=False, desc="Evaluation").attach(rotated_evaluator)

        flipped_evaluator = create_supervised_evaluator(
            L=config.nsym, model=model, metrics=flipped_metrics, device=DEVICE, prepare_batch=prepare_batch
        )
        ProgressBar(persist=False, desc="Evaluation").attach(flipped_evaluator)

        # track training with wandb
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, classic_evaluator, classic_test_loader)
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, rotated_evaluator, rotated_test_loader)
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, flipped_evaluator, flipped_test_loader)

        trainer.run(train_loader, max_epochs=EPOCHS)


if __name__ == "__main__":
    config = build_config()
    for _ in range(MAX_ITER):
        train(config)
