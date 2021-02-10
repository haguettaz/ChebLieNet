import sys
from time import time

import numpy as np
import pykeops
import torch
from gechebnet.graph.graph import SE2GEGraph, SO3GEGraph
from tqdm import tqdm

DEVICE = torch.device("cuda")


def compile_graphs(knn):
    graph = SE2GEGraph(nx=10, ny=10, ntheta=10, knn=knn, device=DEVICE)

    if graph.num_edges < graph.num_nodes:
        raise ValueError(f"An error occured during the computation of the SE2 group equivariant {knn}-NN graph")

    # graph = SO3GEGraph(nsamples=100, nalpha=10, knn=knn, device=DEVICE)

    # if graph.num_edges < graph.num_nodes:
    #     raise ValueError(
    #         f"An error occured during the computation of the SO3 group equivariant {knn}-NN graph"
    #     )


def test_speed(knn, iter=10):
    computation_times = []
    for _ in tqdm(range(iter), file=sys.stdout):
        start = time()
        graph = SE2GEGraph(nx=50, ny=50, ntheta=12, knn=knn, device=DEVICE)
        end = time()
        computation_times.append(end - start)
    print(
        f"{knn}-NN SE2GEGraph with {graph.num_nodes} nodes and {graph.num_edges} edges took",
        f"{np.mean(computation_times)}s +/- {np.std(computation_times)}s to compute",
    )

    # computation_times = []
    # for _ in tqdm(range(iter), file=sys.stdout):
    #     start = time()
    #     graph = SO3GEGraph(nsamples=2500, nalpha=12, knn=knn, device=DEVICE)
    #     end = time()
    #     computation_times.append(end - start)
    # print(
    #     f"{knn}-NN SO3GEGraph with {graph.num_nodes} nodes and {graph.num_edges} edges took",
    #     f"{np.mean(computation_times)}s +/- {np.std(computation_times)}s to compute",
    # )


if __name__ == "__main__":
    pykeops.clean_pykeops()

    for knn in [4, 8, 16, 32, 64]:
        compile_graphs(knn)

    for knn in [4, 8, 16, 32, 64]:
        test_speed(knn)
