import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


class CurriculumTrainer:
    """Trainer for curriculum learning.

    This class is responsible for training the model using the curriculum learning process.
    The training process is defined by the user by extending this class and implementing the
    run method.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer to be used for training.
        loss_module (_Loss): The loss module to be used for training.
        train_data_loader (DataLoader): The data loader for the training data.
        validation_data_loader (DataLoader): The data loader for the validation data.
        parameters (dict): A dictionary containing the parameters for the training process (provided by the scheduler for each curriculum step)
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss: _Loss,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        curriculum_step: int,
        hyperparameters: dict,
        **kwargs,
    ) -> None:
        # Model, optimizer, loss module and data loader
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.loss: _Loss = loss
        self.train_data_loader: DataLoader = train_data_loader
        self.validation_data_loader: DataLoader = validation_data_loader

        # Parameters
        self.curriculum_step: int = curriculum_step
        self.hyperparameters: dict = hyperparameters

        # Other
        self.device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.kwargs = kwargs

    def run(self, **kwargs) -> None:
        """Runs the training process."""
        raise NotImplementedError("run method is not implemented")
