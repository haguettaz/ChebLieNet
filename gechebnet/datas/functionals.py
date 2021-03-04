# coding=utf-8

import numpy as np
import torch
from PIL import Image


def _is_numpy(input):
    """
    Check if input is a numpy object.

    Args:
        input (:obj:): input.

    Returns:
        (bool): True if input is a numpy object.
    """
    return isinstance(input, np.ndarray)


def _is_pil_image(input):
    """
    Check if input is a ``PIL Image``.

    Args:
        input (:obj:): input.

    Returns:
        (bool): True if input is a ``PIL Image``.
    """
    return isinstance(input, Image.Image)


def pil_to_tensor(input):
    """
    Convert a ``PIL Image`` to a tensor of the same type.
    This function does not support torchscript.

    Args:
        input (`PIL.Image.Image`): input PIL image to be converted to tensor.

    Returns:
        (torch.Tensor): output tensor.
    """
    if not _is_pil_image(input):
        raise TypeError("input should be PIL Image. Got {}".format(type(input)))

    default_float_dtype = torch.get_default_dtype()

    # handle PIL Image
    if input.mode == "I":
        output = torch.from_numpy(np.array(input, np.int32, copy=False))
    elif input.mode == "I;16":
        output = torch.from_numpy(np.array(input, np.int16, copy=False))
    elif input.mode == "F":
        output = torch.from_numpy(np.array(input, np.float32, copy=False))
    elif input.mode == "1":
        output = 255 * torch.from_numpy(np.array(input, np.uint8, copy=False))
    else:
        output = torch.ByteTensor(torch.ByteStorage.from_buffer(input.tobytes()))

    output = output.view(input.size[1], input.size[0], len(input.getbands()))
    # put it from HWC to CHW format
    output = output.permute((2, 0, 1)).contiguous()
    if isinstance(output, torch.ByteTensor):
        return output.to(dtype=default_float_dtype).div(255)
    else:
        return output
