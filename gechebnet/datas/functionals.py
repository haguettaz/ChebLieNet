import numpy as np
import torch
from PIL import Image


def _is_numpy(item):
    """
    Check if item is a numpy object.

    Args:
        item (:obj:): item.

    Returns:
        (bool): True if item is a numpy object.
    """
    return isinstance(item, np.ndarray)


def _is_pil_image(item):
    """
    Check if item is a ``PIL Image``.

    Args:
        item (:obj:): item.

    Returns:
        (bool): True if item is a ``PIL Image``.
    """
    return isinstance(item, Image.Image)


def pil_to_tensor(item):
    """
    Convert a ``PIL Image`` to a tensor of the same type.
    This function does not support torchscript.

    Args:
        item (:obj:``PIL Image``): image to be converted to tensor.

    Returns:
        (Tensor): converted image.
    """
    if not _is_pil_image(item):
        raise TypeError("item should be PIL Image. Got {}".format(type(item)))

    default_float_dtype = torch.get_default_dtype()

    # handle PIL Image
    if item.mode == "I":
        img = torch.from_numpy(np.array(item, np.int32, copy=False))
    elif item.mode == "I;16":
        img = torch.from_numpy(np.array(item, np.int16, copy=False))
    elif item.mode == "F":
        img = torch.from_numpy(np.array(item, np.float32, copy=False))
    elif item.mode == "1":
        img = 255 * torch.from_numpy(np.array(item, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))

    img = img.view(item.size[1], item.size[0], len(item.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype).div(255)
    else:
        return img
