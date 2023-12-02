from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


class CurriculumTrainer:
    """Trainer for curriculum learning.

    This class is responsible for training the model using the curriculum learning process.
    The training process is defined by the user by extending this class and implementing the
    run method. The closure method is also expected to be implemented by the user, which is
    used by the optimizer to calculate loss terms, propagate gradients and return the loss value.
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
        logging_path: Optional[str] = None,
        device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        **kwargs,
    ) -> None:
        """Initializes the curriculum trainer.

        Args:
            model (nn.Module): The model to be trained.
            optimizer (Optimizer): The optimizer to be used.
            loss (_Loss): The loss module (or function) to be used.
            train_data_loader (DataLoader): The data loader for the training data.
            validation_data_loader (DataLoader): The data loader for the validation data.
            curriculum_step (int): The current curriculum step.
            hyperparameters (dict): Hyperparameters of the curriculum learning process.
            logging_path (str, optional): Path to the logging directory. Defaults to None.
            device (str, optional): On which device the process should be run. Defaults to "cuda" if available, otherwise "cpu".
        """
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
        self.logging_path: Optional[str] = logging_path
        self.device: str = device
        self.kwargs = kwargs

    def closure(self) -> torch.Tensor:
        """Closure for the optimizer.
        This method is intended to calculate loss terms, propagate gradients and return the loss value.
        Compare: https://pytorch.org/docs/stable/optim.html
        """
        raise NotImplementedError("closure method is not implemented")

    def run(self, **kwargs) -> None:
        """Runs the training process."""
        raise NotImplementedError("run method is not implemented")

    def _optimize(self) -> torch.Tensor:
        """Internal method for optimizing the model.
        This method helps to correctly handle the optimizer and the closure method.

        Returns:
            torch.Tensor: The loss value.
        """
        if type(self.optimizer).__name__ in ["LBFGS"]:
            loss = self.optimizer.step(self.closure)
        else:
            loss = self.closure()
            self.optimizer.step()

        return loss
