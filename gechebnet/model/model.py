from .chebnet import ChebNet


def get_model(name, model_args, device=None):
    """
    [summary]

    Args:
        name ([type]): [description]
        model_args ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    device = device or torch.device("cpu")

    if name == "chebnet":
        model = ChebNet(**model_args)

    return model.to(device)
