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
        loss_module: _Loss,
        scheduler: CurriculumScheduler,
        trainer: Type[CurriculumTrainer],
        evaluator: Type[CurriculumEvaluator],
        device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        logging: bool = False,
        **kwargs,
    ) -> None:
        # Model, optimizer, loss module and data loader
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.loss_module: _Loss = loss_module

        # Curriculum learning components
        self.scheduler: CurriculumScheduler = scheduler
        self.trainer: Type[CurriculumTrainer] = trainer
        self.evaluator: Type[CurriculumEvaluator] = evaluator

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

        self.init_logging(overview=self.scheduler.get_parameters(overview=True))

        while self.scheduler.has_next():
            # Update scheduler
            self.scheduler.next()

            # Get data loader and parameters for current curriculum step
            tdata_loader = self.scheduler.get_train_data_loader()
            vdata_loader = self.scheduler.get_validation_data_loader()
            edata_loader = self.scheduler.get_test_data_loader()
            parameters = self.scheduler.get_parameters()

            # Start training
            trainer = self.trainer(
                self.model,
                self.optimizer,
                self.loss_module,
                tdata_loader,
                vdata_loader,
                parameters,
            )
            trainer.run(**self.kwargs)

            # Evaluate model
            evaluator = self.evaluator(self.model, edata_loader, **parameters)
            evaluator.run(**self.kwargs)

            # Logging
            self.curriculum_step_logging(model=self.model, parameters=parameters)

        self.end_logging(model=self.model, parameters=parameters)
