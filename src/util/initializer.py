# Module containing helper functions for the initialization of models, optimizers, etc.

from model import ConvectionPINNModel
from torch import optim, nn


def initialize_model(
    config: dict,
) -> nn.Module:
    """Initializes a model.

    Args:
        config (dict): The configuration dictionary. Needs to contain the following keys:
            - name (str): The name of the model. Depending on this name, different parameters are expected in the configuration dictionary.

    Returns:
        nn.Module: The initialized model.
    """
    if config["name"] == "ConvectionPINNModel":
        model = ConvectionPINNModel(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
        )
    else:
        raise NotImplementedError(f"Model {config['name']} not implemented.")

    return model


def initialize_optimizer(
    config: dict,
    model: nn.Module,
) -> optim.Optimizer:
    """Initializes an optimizer.

    Args:
        config (dict): The configuration dictionary. Needs to contain the following keys:
            - name (str): The name of the optimizer. Depending on this name, different parameters are expected in the configuration dictionary.
        model (nn.Module): The model to be optimized.

    Returns:
        Optimizer: The initialized optimizer.
    """
    if config["name"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    elif config["name"] == "LBFGS":
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=config["lr"],
            history_size=config["history_size"],
            max_iter=config["max_iter"],
        )
    elif config["name"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=config["nesterov"],
        )
    else:
        raise NotImplementedError(f"Optimizer {config['name']} not implemented.")

    return optimizer
