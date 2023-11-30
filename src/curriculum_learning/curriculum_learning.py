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
        self.kwargs = kwargs

    def run(self) -> None:
        """Runs the curriculum learning process."""
        while self.scheduler.has_next():
            # Update scheduler
            self.scheduler.next()

            # Print current curriculum step
            print(
                f"Curriculum step: {self.scheduler.curriculum_step}/{self.scheduler.max_iter}"
            )

            # Get data loader and parameters for current curriculum step
            data_loader = self.scheduler.get_data_loader()
            edata_loader = self.scheduler.get_eval_data_loader()
            parameters = self.scheduler.get_parameters()

            # Start training
            trainer = self.trainer(
                self.model, self.optimizer, self.loss_module, data_loader, **parameters
            )
            trainer.run(**self.kwargs)

            # Evaluate model
            evaluator = self.evaluator(self.model, edata_loader, **parameters)
            evaluator.run(**self.kwargs)

            # Save model
            torch.save(
                self.model.state_dict(),
                f"tmp/model_{self.scheduler.curriculum_step}.pt",
            )
