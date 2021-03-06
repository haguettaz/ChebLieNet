import argparse
import os

import torch
import wandb
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import AveragePrecision
from ignite.engine import Events
from ignite.metrics import Accuracy, ConfusionMatrix, Fbeta, Loss, Precision, Recall
from ignite.metrics.confusion_matrix import cmAccuracy, mIoU
from torch.nn.functional import nll_loss
from torch.optim import Adam

from cheblienet.datas.dataloaders import get_test_loader, get_train_val_loaders
from cheblienet.engines.engines import create_supervised_evaluator, create_supervised_trainer
from cheblienet.engines.utils import PerClassAccuracy, output_transform_mAP, prepare_batch, wandb_log
from cheblienet.graphs.graphs import S2GEGraph, SO3GEGraph
from cheblienet.nn.models.chebnets import SO3GEUChebNet
from cheblienet.nn.models.utils import capacity


def build_config(anisotropic):
    """
    Gets training configuration.

    Args:
        pool (bool): if True, use a pooling layers.

    Returns:
        (dict): configuration dictionnary.
    """

    if not anisotropic:
        return {
            "kernel_size": 3,
            "eps": 1.0,
            "K": 8,
            "nalpha": 1,
            "xi_0": 1.0,
            "xi_1": 1.0,
            "xi_2": 1.0,
            "xi_3": 1.0,
            "xi_4": 1.0,
            "xi_5": 1.0,
        }

    return {
        "kernel_size": 3,
        "eps": 0.1,
        "K": 16,
        "nalpha": 6,
        "xi_0": 10.0 / 12,
        "xi_1": 10.0 / 42,
        "xi_2": 10.0 / 162,
        "xi_3": 10.0 / 642,
        "xi_4": 10.0 / 2562,
        "xi_5": 10.0 / 10242,
    }


def train(config=None):
    """
    U-net-like model training on ClimateNet for the segmentation of extreme meteorogolical events.

    Args:
        config (dict, optional): configuration dictionnary. Defaults to None.
    """

    # Initialize a new wandb run
    with wandb.init(config=config, project="cheblienet"):

        config = wandb.config
        wandb.log({"dataset": "artc"})
        wandb.log(vars(args))

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Load model and optimizer
        if args.anisotropic:
            # anisotropic kernels
            graph_lvl0 = SO3GEGraph(
                size=[12, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_0),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl1 = SO3GEGraph(
                size=[42, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_1),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl2 = SO3GEGraph(
                size=[162, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_2),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl3 = SO3GEGraph(
                size=[642, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_3),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl4 = SO3GEGraph(
                size=[2562, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_4),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl5 = SO3GEGraph(
                size=[10242, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_5),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

        else:
            # isotropic kernels
            graph_lvl0 = S2GEGraph(
                size=[12, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_0),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl1 = S2GEGraph(
                size=[42, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_1),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl2 = S2GEGraph(
                size=[162, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_2),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl3 = S2GEGraph(
                size=[642, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_3),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl4 = S2GEGraph(
                size=[2562, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_4),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

            graph_lvl5 = S2GEGraph(
                size=[10242, config.nalpha],
                sigmas=(1.0, config.eps, config.xi_5),
                K=config.K,
                path_to_graph=args.path_to_graph,
            )

        output_graph = S2GEGraph(
            size=[10242, 1],
            sigmas=(1.0, 1.0, 1.0),
            K=config.K,
            path_to_graph=args.path_to_graph,
        )

        # Loads group equivariant Chebnet
        model = SO3GEUChebNet(
            16,
            3,
            config.kernel_size,
            graph_lvl0,
            graph_lvl1,
            graph_lvl2,
            graph_lvl3,
            graph_lvl4,
            graph_lvl5,
            output_graph,
            args.reduction,
            args.expansion,
        ).to(device)

        wandb.log({"capacity": capacity(model)})

        optimizer = Adam(model.parameters(), lr=args.lr)

        # Load dataloaders
        train_loader, _ = get_train_val_loaders(
            "artc",
            batch_size=args.batch_size,
            num_layers=config.nalpha,
            val_ratio=0.0,
            path_to_data=args.path_to_data,
        )

        test_loader = get_test_loader(
            "artc", batch_size=args.batch_size, num_layers=config.nalpha, path_to_data=args.path_to_data
        )

        # Load engines
        trainer = create_supervised_trainer(
            graph=graph_lvl5,
            model=model,
            optimizer=optimizer,
            loss_fn=nll_loss,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Training").attach(trainer)

        # confusion matrix based metrics
        cm = ConfusionMatrix(num_classes=3)
        miou = mIoU(cm)
        miou_wo_bg = mIoU(cm, ignore_index=0)

        # f1 score
        f1 = Fbeta(1, precision=Precision(average=False), recall=Recall(average=False))

        # per class accuracies
        acc_bg = PerClassAccuracy(0)
        acc_ar = PerClassAccuracy(1)
        acc_tc = PerClassAccuracy(2)
        # acc_bg = Accuracy(output_transform=lambda batch: output_transform_accuracy(batch, 0))
        # acc_ar = Accuracy(output_transform=lambda batch: output_transform_accuracy(batch, 1))
        # acc_tc = Accuracy(output_transform=lambda batch: output_transform_accuracy(batch, 2))

        # mean average precision
        mean_average_precision = AveragePrecision(output_transform=output_transform_mAP)

        # loss
        loss = Loss(nll_loss)

        metrics = {
            "test_F1": f1,
            "test_mIoU": miou,
            "test_mIoU_bg": miou_wo_bg,
            "test_loss": loss,
            "test_acc_bg": acc_bg,
            "test_acc_ar": acc_ar,
            "test_acc_tc": acc_tc,
            "test_mAP": mean_average_precision,
        }

        evaluator = create_supervised_evaluator(
            graph=graph_lvl5,
            model=model,
            metrics=metrics,
            device=device,
            prepare_batch=prepare_batch,
        )
        ProgressBar(persist=False, desc="Evaluation").attach(evaluator)

        _ = trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log, evaluator, test_loader)

        # Launchs training
        trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_graph", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--num_experiments", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--anisotropic", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reduction", type=str)
    parser.add_argument("--expansion", type=str)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    config = build_config(anisotropic=args.anisotropic)

    for _ in range(args.num_experiments):
        train(config)
