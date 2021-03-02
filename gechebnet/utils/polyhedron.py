"""
the number of vertices is given by V = 10T + 2 where the triangulation number T is given by
T = b² + bc + c² with c = 0 and b = 2^lvl for icosahedral subdivision.
"""

import math
from typing import Tuple

import torch
from torch import FloatTensor, LongTensor

ICOSAHEDRON_VERTICES = torch.tensor(
    [
        [(1 + math.sqrt(5)) / 2, 1.0, 0.0],
        [-(1 + math.sqrt(5)) / 2, 1.0, 0.0],
        [(1 + math.sqrt(5)) / 2, -1.0, 0.0],
        [-(1 + math.sqrt(5)) / 2, -1.0, 0.0],
        [1.0, 0.0, (1 + math.sqrt(5)) / 2],
        [1.0, 0.0, -(1 + math.sqrt(5)) / 2],
        [-1.0, 0.0, (1 + math.sqrt(5)) / 2],
        [-1.0, 0.0, -(1 + math.sqrt(5)) / 2],
        [0.0, (1 + math.sqrt(5)) / 2, 1.0],
        [0.0, -(1 + math.sqrt(5)) / 2, 1.0],
        [0.0, (1 + math.sqrt(5)) / 2, -1.0],
        [0.0, -(1 + math.sqrt(5)) / 2, -1.0],
    ]
)
ICOSAHEDRON_FACES = torch.tensor(
    [
        [0, 2, 4],
        [0, 2, 5],
        [0, 4, 8],
        [0, 8, 10],
        [0, 5, 10],
        [1, 3, 6],
        [1, 3, 7],
        [1, 6, 8],
        [1, 7, 10],
        [1, 8, 10],
        [2, 4, 9],
        [2, 5, 11],
        [2, 9, 11],
        [3, 7, 11],
        [3, 9, 11],
        [3, 6, 9],
        [4, 6, 9],
        [4, 6, 8],
        [5, 7, 10],
        [5, 7, 11],
    ]
)


class SphericalPolyhedron:
    """
    Symbolic class to represent a spherical polyhedron for uniformly sampling on S2.
    """

    def __init__(self, level: int):
        """
        Initialization

        Args:
            level (int): subdivision level of the polyhedron sampling.
        """

        vertices, faces = ICOSAHEDRON_VERTICES, ICOSAHEDRON_FACES

        vertices /= torch.norm(vertices, dim=1, keepdim=True)

        for _ in range(level):
            sub_faces = torch.stack((faces[:, [0, 1, 2]], faces[:, [2, 0, 1]], faces[:, [1, 2, 0]]), dim=1)
            vertices_1 = vertices[sub_faces[:, :, [0, 1]]].mean(dim=2)
            vertices_1 /= torch.norm(vertices_1, dim=2, keepdim=True)
            vertices_2 = vertices[sub_faces[:, :, [1, 2]]].mean(dim=2)
            vertices_2 /= torch.norm(vertices_2, dim=2, keepdim=True)

            N = vertices.shape[0]
            NV1 = vertices_1.shape[0] * vertices_1.shape[1]
            NV2 = vertices_2.shape[0] * vertices_1.shape[1]

            sub_faces[:, :, 0] = torch.arange(N, N + NV1).reshape(-1, 3)
            sub_faces[:, :, 2] = torch.arange(N + NV1, N + NV1 + NV2).reshape(-1, 3)

            faces = torch.cat((sub_faces, torch.arange(N, N + NV1).reshape(-1, 1, 3)), dim=1).reshape(-1, 3)
            vertices = torch.cat((vertices, vertices_1.reshape(-1, 3), vertices_2.reshape(-1, 3)), dim=0)

        self.level = level
        self.vertices = vertices.unique(dim=0)
