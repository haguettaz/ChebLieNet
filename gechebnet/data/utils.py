import random


def split_data_list(data_list, ratio=0.1):
    """
    [summary]

    Args:
        data_list ([type]): [description]
        ratio (float, optional): [description]. Defaults to 0.1.

    Returns:
        [type]: [description]
    """
    data_list_size = len(data_list)
    split = int(data_list_size * ratio)

    random.shuffle(data_list)

    return data_list[split:], data_list[:split]
