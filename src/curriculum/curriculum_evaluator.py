from typing import Optional

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


class CurriculumEvaluator:
    """Base class for curriculum evaluator.

    This class is responsible for evaluating the model using the curriculum learning process.
    The evaluation process is defined by the user by extending this class and implementing the
    run method.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: _Loss,
        data_loader: DataLoader,
        curriculum_step: int,
        hyperparameters: dict,
        logging_path: Optional[str] = None,
        logging_dict: dict = None,
        device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        **kwargs
    ) -> None:
        """Initializes the curriculum evaluator.

        Args:
            model (nn.Module): The model to be evaluated.
            loss (_Loss): The loss module (or function) to be used.
            data_loader (DataLoader): The data loader to be used.
            curriculum_step (int): The current curriculum step.
            hyperparameters (dict): Hyperparameters of the curriculum learning process.
        """
        # Model, optimizer, loss module and data loader
        self.model: nn.Module = model
        self.loss: _Loss = loss
        self.data_loader: DataLoader = data_loader

        # Parameters
        self.curriculum_step: int = curriculum_step
        self.hyperparameters: dict = hyperparameters

        # Other
        self.logging_path: Optional[str] = logging_path
        self.logging_dict: dict = logging_dict

        # Other
        self.device: str = device
        self.kwargs = kwargs

    def run(self) -> None:
        """Runs the evaluation process."""
        raise NotImplementedError("run method is not implemented")
