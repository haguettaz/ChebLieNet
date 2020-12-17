from torch.optim import SGD, Adam


def get_optimizer(model, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer
