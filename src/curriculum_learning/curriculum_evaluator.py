import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


class CurriculumEvaluator:
    """Evaluator class for the curriculum learning process.

    This class is responsible for evaluating the model using the curriculum learning process.
    The training process is defined by the user by extending this class and implementing the
    run method.

    Args:
        model (nn.Module): model to train
        loss (_Loss): loss module used for training
        data_loader (DataLoader): data loader to use
    """

    def __init__(
        self,
        model: nn.Module,
        loss: _Loss,
        data_loader: DataLoader,
        curriculum_step: int,
        hyperparameters: dict,
        **kwargs
    ) -> None:
        # Model, optimizer, loss module and data loader
        self.model: nn.Module = model
        self.loss: _Loss = loss
        self.data_loader: DataLoader = data_loader

        # Parameters
        self.curriculum_step: int = curriculum_step
        self.hyperparameters: dict = hyperparameters

        # Other
        self.device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.kwargs = kwargs

    def run(self) -> None:
        """Runs the evaluation process."""
        raise NotImplementedError("run method is not implemented")
