from typing import Type

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from .curriculum_evaluator import CurriculumEvaluator
from .curriculum_scheduler import CurriculumScheduler
from .curriculum_trainer import CurriculumTrainer


class CurriculumLearning:
    """Base class for curriculum learning.

    This class is responsible for the curriculum learning process.

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss: _Loss,
        scheduler: CurriculumScheduler,
        trainer: Type[CurriculumTrainer],
        evaluator: Type[CurriculumEvaluator],
        hyperparameters: dict,
        device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        logging: bool = False,
        **kwargs,
    ) -> None:
        # Model, optimizer, loss module and data loader
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.loss: _Loss = loss

        # Curriculum learning components
        self.scheduler: CurriculumScheduler = scheduler
        self.trainer: Type[CurriculumTrainer] = trainer
        self.evaluator: Type[CurriculumEvaluator] = evaluator

        # Hyperparameters
        self.hyperparameters: dict = hyperparameters

        # Other
        self.device: str = device
        self.logging: bool = logging
        self.kwargs = kwargs

    def init_logging(self, **kwargs) -> None:
        """Initial logging, before the curriculum learning process starts."""
        pass

    def curriculum_step_logging(self, **kwargs) -> None:
        """Logging for each curriculum step."""
        pass

    def end_logging(self, **kwargs) -> None:
        """Logging after the curriculum learning process ends."""
        pass

    def run(self) -> None:
        """Runs the curriculum learning process."""

        self.init_logging()

        while self.scheduler.has_next():
            # Update scheduler
            self.scheduler.next()

            # Get data loader and parameters for current curriculum step
            tdata_loader = self.scheduler.get_train_data_loader()
            vdata_loader = self.scheduler.get_validation_data_loader()
            edata_loader = self.scheduler.get_test_data_loader()

            # Start training
            trainer = self.trainer(
                self.model,
                self.optimizer,
                self.loss,
                tdata_loader,
                vdata_loader,
                self.scheduler.curriculum_step,
                self.hyperparameters,
            )
            trainer.run(**self.kwargs)

            # Evaluate model
            evaluator = self.evaluator(
                self.model,
                self.loss,
                edata_loader,
                self.scheduler.curriculum_step,
                self.hyperparameters,
            )
            evaluator.run(**self.kwargs)

            # Logging
            self.curriculum_step_logging(model=self.model)

        self.end_logging(model=self.model)
