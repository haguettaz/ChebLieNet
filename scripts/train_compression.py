import math
import os

import torch
import torch.nn.functional as F
import wandb
from gechebnet.data.dataloader import get_data_list_mnist, get_data_loader
from gechebnet.data.dataset import download_mnist, download_rotated_mnist
from gechebnet.data.utils import split_data_list
from gechebnet.graph.graph import GraphData
from gechebnet.graph.utils import GaussianKernel
from gechebnet.model.chebnet import ChebNet
from gechebnet.utils import prepare_batch, wandb_log
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from torch.optim import SGD, Adam

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 1
OUTPUT_DIM = 10


# def build_sweep_config():
#     sweep_config = {"method": "random", "metric": {"name": "val_mnist_acc", "goal": "maximize"}}

#     parameters_dict = {
#         "batch_size": {"value": },
#         "weight_thresh": {"value": },
#         "edge_red": {"value": },
#         "epochs": {"value": 20},
#         "eps": {"value": },
#         "hidden_dim": {"value": 16},
#         "K": {"value": },
#         "learning_rate": {"value": },
#         "nx1": {"value": 28},
#         "nx2": {"value": 28},
#         "nx3": {"values": [2, 3, 6]},
#         "optimizer": {"value": },
#         "weight_sigma": {"value": },
#         "val_ratio": {"value": 0.2},
#         "xi": {"value": },
#         "compression_type": {"values": ["node", "edge"]},
#         "compression_rate": {"values": [0.2, 0.5, 0.9]}
#     }
#     sweep_config["parameters"] = parameters_dict

#     return sweep_config


def build_data_loaders(nx1, nx2, nx3, sigma1, sigma2, sigma3, weight_thresh, weight_sigma, compression_type, kappa, batch_size, val_ratio):

    graph_data = GraphData(
        grid_size=(nx1, nx2),
        num_layers=nx3,
        self_loop=True,
        static_compression=(compression_type, kappa),
        weight_kernel=GaussianKernel(weight_thresh, weight_sigma),
        sigmas=(sigma1, sigma2, sigma3),
    )

    processed_path = download_mnist(DATA_PATH)

    train_data_list, val_data_list = split_data_list(get_data_list_mnist(graph_data, processed_path, train=True), ratio=val_ratio)

    train_loader = get_data_loader(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader = get_data_loader(val_data_list, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = Adam(network.parameters(), lr=learning_rate)
    return optimizer


def build_network(K, nx3, input_dim, output_dim, hidden_dim, edge_red, device=None):

    device = device or torch.device("cpu")

    model = ChebNet(K, nx3, input_dim, output_dim, hidden_dim, edge_red)

    return model.to(device)


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader, val_loader = build_data_loaders(
            nx1=config.nx1,
            nx2=config.nx2,
            nx3=config.nx3,
            sigma1=config.xi / config.eps,
            sigma2=config.xi,
            sigma3=1.0,
            weight_thresh=config.weight_thresh,
            weight_sigma=config.weight_sigma,
            compression_type=config.compression_type,
            kappa=config.compression_rate,
            batch_size=config.batch_size,
            val_ratio=config.val_ratio,
        )

        network = build_network(config.K, config.nx3, INPUT_DIM, OUTPUT_DIM, config.hidden_dim, config.edge_red, DEVICE)

        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        loss_fn = F.nll_loss
        metrics = {"val_mnist_acc": Accuracy(), "val_mnist_loss": Loss(loss_fn)}

        # create ignite's engines
        trainer = create_supervised_trainer(network, optimizer, loss_fn, DEVICE, prepare_batch=prepare_batch)
        ProgressBar(persist=False, desc="Training").attach(trainer)

        evaluator = create_supervised_evaluator(network, metrics, DEVICE, prepare_batch=prepare_batch)
        ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        # track training with wandb
        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, val_loader)

        # save best model
        trainer.run(train_loader, max_epochs=config.epochs)


if __name__ == "__main__":
    sweep_config = build_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="gechebnet")
    wandb.agent(sweep_id, train, count=50)
