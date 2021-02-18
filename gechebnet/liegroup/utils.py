import math
from typing import Tuple

import torch
from torch import FloatTensor

from ..utils import mod


def weighted_norm(input, Re):
    Re = Re.to(input.device)
    return torch.matmul(torch.matmul(input.transpose(1, 2), Re), input).squeeze()
