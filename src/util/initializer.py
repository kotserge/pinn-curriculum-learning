# Module containing helper functions for the initialization of models, optimizers, etc.

from model import ConvectionPINNModel
from loss import ConvectionMSEPDELoss

from torch import optim, nn
from torch.utils.data import Dataset, Sampler, RandomSampler
from torch.nn.modules.loss import _Loss


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
    assert "name" in config, "name parameter is required for model initialization"

    if config["name"] == "ConvectionPINNModel":
        assert (
            "input_dim" in config
        ), "input_dim parameter is required for ConvectionPINNModel"
        assert (
            "hidden_dim" in config
        ), "hidden_dim parameter is required for ConvectionPINNModel"

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
    assert "name" in config, "name parameter is required for optimizer initialization"

    if config["name"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"] if "lr" in config else 1e-3,
            betas=config["betas"] if "betas" in config else (0.9, 0.999),
            eps=config["eps"] if "eps" in config else 1e-8,
            weight_decay=config["weight_decay"] if "weight_decay" in config else 0.0,
        )

    elif config["name"] == "LBFGS":
        assert (
            "line_search_fn" not in config
            or "line_search_fn" in config
            and config["line_search_fn"] == "strong_wolfe"
        ), "line_search_fn parameter must be 'strong_wolfe' for LBFGS optimizer initialization or not set"

        optimizer = optim.LBFGS(
            model.parameters(),
            lr=config["lr"] if "lr" in config else 1.0,
            max_iter=config["max_iter"] if "max_iter" in config else 20,
            max_eval=config["max_eval"] if "max_eval" in config else None,
            tolerance_grad=config["tolerance_grad"]
            if "tolerance_grad" in config
            else 1e-7,
            tolerance_change=config["tolerance_change"]
            if "tolerance_change" in config
            else 1e-9,
            history_size=config["history_size"] if "history_size" in config else 100,
            line_search_fn=config["line_search_fn"]
            if "line_search_fn" in config
            else None,
        )

    elif config["name"] == "SGD":
        assert (
            "lr" in config
        ), "lr parameter is required for SGD optimizer initialization"

        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"] if "momentum" in config else 0.0,
            dampening=config["dampening"] if "dampening" in config else 0.0,
            weight_decay=config["weight_decay"] if "weight_decay" in config else 0.0,
            nesterov=config["nesterov"] if "nesterov" in config else False,
        )
    else:
        raise NotImplementedError(f"Optimizer {config['name']} not implemented.")

    return optimizer


def initialize_loss(
    config: dict,
    **kwargs,
) -> _Loss:
    """Initializes a loss module.

    Args:
        config (dict): The configuration dictionary. Needs to contain the following keys:
            - name (str): The name of the loss module. Depending on this name, different parameters are expected in the configuration dictionary.

    Returns:
        _Loss: The initialized loss module.
    """
    assert "name" in config, "name parameter is required for loss initialization"

    if config["name"] == "ConvectionMSEPDELoss":
        assert (
            "curriculum_step" in kwargs
        ), "curriculum_step parameter is required for ConvectionMSEPDELoss"
        assert (
            "convection" in config
        ), "convection parameter is required for ConvectionMSEPDELoss"
        assert kwargs["curriculum_step"] < len(
            config["convection"]
        ), f"Convection parameter for curriculum step {kwargs['curriculum_step']} not found"
        assert (
            "regularization" in config
        ), "regularization parameter is required for ConvectionMSEPDELoss"
        assert "model" in kwargs, "model parameter is required for ConvectionMSEPDELoss"

        loss = ConvectionMSEPDELoss(
            convection=config["convection"][kwargs["curriculum_step"]],
            regularization=config["regularization"],
            model=kwargs["model"],
        )
    else:
        raise NotImplementedError(f"Loss module {config['name']} not implemented.")

    return loss


def initialize_sampler(
    config: dict,
    dataset: Dataset,
) -> Sampler:
    """Initializes a sampler.

    Args:
        config (dict): The configuration dictionary. Needs to contain the following keys:
            - name (str): The name of the sampler. Depending on this name, different parameters are expected in the configuration dictionary.
        dataset (Dataset): The dataset to be sampled.

    Returns:
        Sampler: The initialized sampler.
    """
    if config["name"] == "RandomSampler":
        assert (
            "replacement" in config
        ), "replacement (bool) parameter is required for RandomSampler"
        assert (
            "num_samples" in config or "percent_samples" in config
        ), "num_samples (int) or percent_samples (float) parameter is required for RandomSampler"
        assert (
            "percent_samples" not in config or "num_samples" not in config
        ), "num_samples and percent_samples parameters are mutually exclusive for RandomSampler"
        assert (
            "percent_samples" in config
            and isinstance(config["percent_samples"], float)
            and 0.0 <= config["percent_samples"] <= 1.0
        ) or (
            "num_samples" in config
            and isinstance(config["num_samples"], int)
            and config["num_samples"] > 0
        ), "percent_samples (float) must be in range [0.0, 1.0] or num_samples (int) must be greater than 0 for RandomSampler"

        num_samples = None
        if "percent_samples" in config:
            num_samples = int(len(dataset) * config["percent_samples"])
        elif "num_samples" in config:
            num_samples = config["num_samples"]

        sampler = RandomSampler(
            data_source=dataset,
            replacement=config["replacement"],
            num_samples=num_samples,
        )
    else:
        raise NotImplementedError(f"Sampler {config['name']} not implemented.")

    return sampler
