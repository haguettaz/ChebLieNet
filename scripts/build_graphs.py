import sys
from time import time

import numpy as np
from gechebnet.graph.graph import SE2GEGraph
from tqdm import tqdm

NX1, NX2, NX3 = 100, 100, 20


def compile_graphs(knn):

    graph = SE2GEGraph(
        grid_size=(NX1, NX2),
        nx3=NX3,
        knn=knn,
    )

    if graph.num_edges < graph.num_nodes:
        raise ValueError(f"An error occured during the computation of the {knn}-NN graph")


def test_speed(knn, iter=50):
    computation_times = []
    for _ in tqdm(range(iter), file=sys.stdout):
        start = time()
        graph = SE2GEGraph(
            grid_size=(NX1, NX2),
            nx3=NX3,
            knn=knn,
        )
        end = time()
        computation_times.append(end - start)

        if graph.num_edges < graph.num_nodes:
            raise ValueError(f"An error occured during the computation of the {knn}-NN graph")

    print(
        f"{knn}-NN SE2GEGraph with {graph.num_nodes} nodes and {graph.num_edges} edges took",
        f"{np.mean(computation_times)}s +/- {np.std(computation_times)}s to compute",
    )


if __name__ == "__main__":
    for knn in [2, 4, 8, 16, 32, 64]:
        # compile_graphs(knn)
        test_speed(knn)
