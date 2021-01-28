import sys
from time import time

import numpy as np
from gechebnet.graph.graph import SE2GEGraph, SO3GEGraph
from tqdm import tqdm


def compile_graphs(knn):
    graph = SE2GEGraph(
        nx=100,
        ny=100,
        ntheta=12,
        knn=knn,
    )

    if graph.num_edges < graph.num_nodes:
        raise ValueError(
            f"An error occured during the computation of the SE2 group equivariant {knn}-NN graph"
        )

    graph = SO3GEGraph(
        nsamples=10000,
        nalpha=12,
        knn=knn,
    )

    if graph.num_edges < graph.num_nodes:
        raise ValueError(
            f"An error occured during the computation of the SO3 group equivariant {knn}-NN graph"
        )


def test_speed(knn, iter=50):
    computation_times = []
    for _ in tqdm(range(iter), file=sys.stdout):
        start = time()
        graph = SE2GEGraph(
            nx=100,
            ny=100,
            ntheta=12,
            knn=knn,
        )
        end = time()
        computation_times.append(end - start)

        if graph.num_edges < graph.num_nodes:
            raise ValueError(
                f"An error occured during the computation of the SE2 group equivariant {knn}-NN graph"
            )

    print(
        f"{knn}-NN SE2GEGraph with {graph.num_nodes} nodes and {graph.num_edges} edges took",
        f"{np.mean(computation_times)}s +/- {np.std(computation_times)}s to compute",
    )


    computation_times = []
    for _ in tqdm(range(iter), file=sys.stdout):
        start = time()
        graph = SO3GEGraph(
            nsamples=10000,
            nalpha=12,
            knn=knn,
        )
        end = time()
        computation_times.append(end - start)

        if graph.num_edges < graph.num_nodes:
            raise ValueError(
                f"An error occured during the computation of the SO3 group equivariant {knn}-NN graph"
            )

    print(
        f"{knn}-NN SO3GEGraph with {graph.num_nodes} nodes and {graph.num_edges} edges took",
        f"{np.mean(computation_times)}s +/- {np.std(computation_times)}s to compute",
    )


if __name__ == "__main__":
    for knn in [2, 4, 8, 16, 32, 64]:
        compile_graphs(knn)
        test_speed(knn)
