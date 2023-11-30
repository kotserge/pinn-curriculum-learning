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
        model (nn.Module): model to train
        optimizer (Optimizer): optimizer to use
        loss_module (_Loss): loss function to use
        data_loader (DataLoader): data loader to use
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_module: _Loss,
        data_loader: DataLoader,
        epochs: int,
        **kwargs,
    ) -> None:
        # Model, optimizer, loss module and data loader
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.loss_module: _Loss = loss_module
        self.data_loader: DataLoader = data_loader

        # Parameters
        self.epochs: int = epochs

        # Other
        self.device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.kwargs = kwargs

    def run(self, **kwargs) -> None:
        """Runs the training process."""
        raise NotImplementedError("run method is not implemented")
