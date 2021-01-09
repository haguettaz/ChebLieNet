import math
from typing import Optional, Tuple

import torch
from pykeops.torch import LazyTensor, Pm

from ..utils import lower, upper


def se2_group_inverse(x: LazyTensor) -> LazyTensor:
    """
    Inverse SE2 group element(s).

    Args:
        x (LazyTensor): group element(s).

    Returns:
        LazyTensor: inverse group element(s).
    """
    return LazyTensor.cat(
        (-x[0] * x[2].cos() - x[1] * x[2].sin(), -x[1] * x[2].cos() + x[0] * x[2].sin(), -x[2]), dim=-1
    )


def se2_group_product(x1: LazyTensor, x2: LazyTensor) -> LazyTensor:
    """
    Multiply SE2 group element(s) according to the group product.

    Args:
        x1 (LazyTensor): group element(s).
        x2 (LazyTensor): group element(s).

    Returns:
        LazyTensor: group element(s).
    """
    return LazyTensor.cat(
        (
            x1[0] + x2[0] * (x1[2]).cos() - x2[1] * (x1[2]).sin(),
            x1[1] + x2[1] * (x1[2]).cos() + x2[0] * (x1[2]).sin(),
            x1[2] + x2[2],
        ),
        dim=-1,
    )


def se2_group_log(x: LazyTensor) -> LazyTensor:
    """
    Log maps SE2 group element(s).

    Args:
        x (LazyTensor): group element(s).

    Returns:
        LazyTensor: tangent bundle element(s).
    """
    # if theta == 0, cot(theta) return 0, we use this implementation detail for the log expression by part
    eps = 1e-5
    return LazyTensor.cat(
        (
            lower(x[2].mod(math.pi, 0.0).abs(), eps) * x[0]
            + upper(x[2].mod(math.pi, 0.0).abs(), eps)
            * 0.5
            * x[2].mod(math.pi, -math.pi / 2)
            * (x[1] + x[0] * ((x[2].mod(math.pi, -math.pi / 2) / 2).cot())),
            lower(x[2].mod(math.pi, 0.0).abs(), eps) * x[1]
            + upper(x[2].mod(math.pi, 0.0).abs(), eps)
            * 0.5
            * x[2].mod(math.pi, -math.pi / 2)
            * (-x[0] + x[1] * ((x[2].mod(math.pi, -math.pi / 2) / 2).cot())),
            x[2].mod(math.pi, -math.pi / 2),
        ),
        dim=-1,
    )


def metric_tensor(sigmas: Tuple[float, float, float], device: torch.device) -> LazyTensor:
    """
    Get metric tensor of anisotropy parameters.

    Args:
        sigmas (Tuple[float, float, float]): anisotropy parameters in the (x,y,theta) directions.
        device (torch.device): computation device.

    Returns:
        LazyTensor: metric tensor.
    """
    return Pm(torch.tensor([*sigmas], device=device))


def sq_riemannian_distance(xi: LazyTensor, xj: LazyTensor, S: LazyTensor) -> LazyTensor:
    """
    Return squared Riemannian distances between group elements.

    Args:
        xi (LazyTensor): group element(s).
        xj (LazyTensor): group element(s).
        S (LazyTensor): metric tensor.

    Returns:
        LazyTensor: squared Riemannian distance(s).
    """

    v = se2_group_log(se2_group_product(se2_group_inverse(xi), xj))

    return v.weightedsqnorm(S)
