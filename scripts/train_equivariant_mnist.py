import math
import os

import torch
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
from torch.nn import NLLLoss
from torch.nn.functional import nll_loss
from torch.optim import SGD, Adam

DATA_PATH = os.path.join(os.environ["TMPDIR"], "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_RUNS = 10

INPUT_DIM = 1
OUTPUT_DIM = 10


# def build_config(name):
#     if name == "isochebnet":
#         config = {
#             "batch_size": {"value": },
#             "weight_thresh": {"value": },
#             "edge_red": {"value": },
#             "epochs": {"value": 20},
#             "eps": {"value": 1.},
#             "hidden_dim": {"value": 16},
#             "K": {"value": },
#             "learning_rate": {"value": },
#             "nx1": {"value": 28},
#             "nx2": {"value": 28},
#             "nx3": {"value": 1},
#             "optimizer": {"value": },
#             "weight_sigma": {"value": },
#             "val_ratio": {"value": 0.2},
#             "xi": {"value": 1. },
#         }

#     elif name == "gechebnet":
#         config = {
#             "batch_size": {"value": },
#             "weight_thresh": {"value": },
#             "edge_red": {"value": },
#             "epochs": {"value": 20},
#             "eps": {"value": },
#             "hidden_dim": {"value": 16},
#             "K": {"value": },
#             "learning_rate": {"value": },
#             "nx1": {"value": 28},
#             "nx2": {"value": 28},
#             "nx3": {"value": },
#             "optimizer": {"value": },
#             "weight_sigma": {"value": },
#             "val_ratio": {"value": 0.2},
#             "xi": {"value": },
#         }

#     return config


def build_data_loaders(nx1, nx2, nx3, sigma1, sigma2, sigma3, weight_thresh, weight_sigma, batch_size, val_ratio):

    graph_data = GraphData(
        grid_size=(nx1, nx2),
        num_layers=nx3,
        self_loop=True,
        weight_kernel=GaussianKernel(weight_thresh, weight_sigma),
        sigmas=(sigma1, sigma2, sigma3),
    )

    mnist_processed_path = download_mnist(DATA_PATH)
    rot_mnist_processed_path = download_rotated_mnist(DATA_PATH)

    train_mnist_data_list, _ = split_data_list(
        get_data_list_mnist(graph_data, mnist_processed_path, train=True), ratio=val_ratio
    )
    train_mnist_loader = get_data_loader(train_data_list, batch_size=batch_size, shuffle=True)

    test_mnist_data_list = get_data_list_mnist(graph_data, mnist_processed_path, train=False)
    test_mnist_loader = get_data_loader(test_mnist_data_list, batch_size=batch_size, shuffle=True)

    test_rot_mnist_data_list = get_data_list_mnist(graph_data, rot_mnist_processed_path, train=False)
    test_rot_mnist_loader = get_data_loader(test_rot_mnist_data_list, batch_size=batch_size, shuffle=True)

    return train_mnist_loader, test_mnist_loader, test_rot_mnist_loader


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
    with wandb.init(project="chebnet", config=config):
        config = wandb.config

        train_mnist_loader, test_mnist_loader, test_rot_mnist_loader = build_data_loaders(
            nx1=config.nx1,
            nx2=config.nx2,
            nx3=config.nx3,
            sigma1=config.xi / config.eps,
            sigma2=config.xi,
            sigma3=1.0,
            weight_thresh=config.weight_thresh,
            weight_sigma=config.weight_sigma,
            batch_size=config.batch_size,
            val_ratio=config.val_ratio,
        )

        network = build_network(config.K, config.nx3, INPUT_DIM, OUTPUT_DIM, config.hidden_dim, config.edge_red, DEVICE)

        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        criterion = NLLLoss()
        mnist_metrics = {"test_mnist_acc": Accuracy(), "test_mnist_loss": Loss(nll_loss)}
        rot_mnist_metrics = {"test_rot_mnist_acc": Accuracy(), "test_rot_mnist_loss": Loss(nll_loss)}

        # create ignite's engines
        mnist_trainer = create_supervised_trainer(network, optimizer, criterion, DEVICE, prepare_batch=prepare_batch)
        mnist_evaluator = create_supervised_evaluator(network, mnist_metrics, DEVICE, prepare_batch=prepare_batch)
        rot_mnist_evaluator = create_supervised_evaluator(
            network, rot_mnist_metrics, DEVICE, prepare_batch=prepare_batch
        )

        # track training with wandb
        _ = mnist_trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, mnist_evaluator, test_mnist_loader)
        _ = mnist_trainer.add_event_handler(
            Events.EPOCH_COMPLETED, wandb_log, rot_mnist_evaluator, test_rot_mnist_loader
        )

        # save best model
        mnist_trainer.run(train_mnist_loader, max_epochs=config.epochs)


if __name__ == "__main__":

    for _ in range(NUM_RUNS):
        config = build_config("isochebnet")
        train(config)

        config = build_config("gechebnet")
        train(config)
