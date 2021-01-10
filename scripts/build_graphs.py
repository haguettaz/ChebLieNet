from time import time

import numpy as np
from gechebnet.graph.graph import HyperCubeGraph

NX1, NX2, NX3 = 100, 100, 20


def compile_graphs(knn):

    graph = HyperCubeGraph(
        grid_size=(NX1, NX2),
        nx3=NX3,
        knn=knn,
    )

    if graph.num_edges < graph.num_nodes:
        raise ValueError(f"An error occured during the computation of the {knn}-NN graph")


def test_speed(knn, iter=50):
    computation_times = np.zeros(iter)
    for i in range(iter):
        start = time()
        graph = HyperCubeGraph(
            grid_size=(NX1, NX2),
            nx3=NX3,
            knn=knn,
        )
        end = time()
        computation_times[i] = end - start

        if graph.num_edges < graph.num_nodes:
            raise ValueError(f"An error occured during the computation of the {knn}-NN graph")

    print(
        f"{knn}-NN HyperCubeGraph with {graph.num_nodes} nodes and {graph.num_edges} edges took",
        f"{computation_times.mean()}s +/- {computation_times.std()}s to compute",
    )


if __name__ == "__main__":
    for knn in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        # compile_graphs(knn)
        test_speed(knn)
