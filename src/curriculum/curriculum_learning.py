from typing import Type, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from .curriculum_evaluator import CurriculumEvaluator
from .curriculum_scheduler import CurriculumScheduler
from .curriculum_trainer import CurriculumTrainer


class CurriculumLearning:
    """Base class for curriculum learning.

    This class is responsible for the curriculum learning process based on the given scheduler, trainer and evaluator.

    It is possible to use the class as is, without any logging. Logging is done by calling the init_logging,
    curriculum_step_logging and end_logging methods. These methods should be implemented by the user by extending
    this class. If logging is not required, the methods can be left empty.
    """

    def __init__(
        self,
        modelzz: Type[nn.Module],
        optimizerzz: Type[Optimizer],
        loss: _Loss,
        schedulerzz: Type[CurriculumScheduler],
        trainerzz: Type[CurriculumTrainer],
        evaluatorzz: Type[CurriculumEvaluator],
        hyperparameters: dict,
        logging_path: Optional[str] = None,
        logging_dict: dict = None,
        device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        **kwargs,
    ) -> None:
        """Initializes the curriculum learning process.

        Args:
            modelzz (nn.Module): The model class to be used. This class needs to be initialized with the hyperparameters during the curriculum learning process.
            optimizerzz (Optimizer): The optimizer class to be used. This class needs to be initialized with the model parameters during the curriculum learning process.
            loss (_Loss): The loss module (or function) to be used.
            scheduler (CurriculumScheduler): The scheduler to be used.
            trainer (Type[CurriculumTrainer]): The trainer class to be used.
            evaluator (Type[CurriculumEvaluator]): The evaluator class to be used.
            hyperparameters (dict): Hyperparameters of the curriculum learning process.
            device (str, optional): On which device the process should be run. Defaults to "cuda" if available, otherwise "cpu".
            logging_path (str, optional): Path to the logging directory. Defaults to None.
        """
        # Model, optimizer, loss module and data loader
        self.modelzz: nn.Module = modelzz
        self.model: nn.Module = None

        self.optimizerzz: Optimizer = optimizerzz
        self.optimizer: Optimizer = None

        self.loss: _Loss = loss

        # Curriculum learning components
        self.schedulerzz: Type[CurriculumScheduler] = schedulerzz
        self.trainerzz: Type[CurriculumTrainer] = trainerzz
        self.evaluatorzz: Type[CurriculumEvaluator] = evaluatorzz

        # Hyperparameters
        self.hyperparameters: dict = hyperparameters
        self.baseline: bool = not hyperparameters["learning"]["curriculum"]

        # Logging
        self.logging_path: Optional[str] = logging_path
        self.logging_dict: dict = logging_dict if logging_dict is not None else {}

        # Other
        self.device: str = device
        self.kwargs = kwargs

        # Preparation for baseline training (i.e. model is reset to initial state after each curriculum step)
        if self.baseline:
            self.init_model_state_dict = self.model.state_dict()
            self.init_optimizer_state_dict = self.optimizer.state_dict()

    def initialize(self, **kwargs) -> None:
        """Function for initialization before the curriculum learning process starts.

        By default the scheduler is  initialized here, and if overriden, the user should
            call super().initialize() or initialize the scheduler manually.
        """

        self.scheduler = self.schedulerzz(
            config=self.hyperparameters,
            **self.kwargs,
        )

    def curriculum_step_preprocessing(self, **kwargs) -> None:
        """Function for preprocessing before each curriculum step.
        By default the model and optimizer are initialized here, and if overriden, the user should
            call super().curriculum_step_preprocessing() or initialize the model and optimizer manually.
        """
        self.trainer = self.trainerzz(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            train_data_loader=self.latest_tdata_loader,
            validation_data_loader=self.latest_vdata_loader,
            curriculum_step=self.scheduler.curriculum_step,
            config=self.hyperparameters,
            device=self.device,
            logging_path=self.logging_path,
            logging_dict=self.logging_dict,
        )

        self.evaluator = self.evaluatorzz(
            model=self.model,
            loss=self.loss,
            data_loader=self.latest_edata_loader,
            curriculum_step=self.scheduler.curriculum_step,
            config=self.hyperparameters,
            device=self.device,
            logging_path=self.logging_path,
            logging_dict=self.logging_dict,
        )

    def curriculum_step_postprocessing(self, **kwargs) -> None:
        """Function for processing after each curriculum step.

        By default the model is reset to the initial state if baseline training is used, and if overriden, the user should
            call super().curriculum_step_postprocessing() or reset the model manually.
        """
        # Reset model to initial state if baseline training is used
        if self.baseline:
            self.model.load_state_dict(self.init_model_state_dict)
            self.optimizer.load_state_dict(self.init_optimizer_state_dict)

    def finalize(self, **kwargs) -> None:
        """Function for finalization after the curriculum learning process ends."""
        pass

    def run(self) -> None:
        """Runs the curriculum learning process.

        This method is responsible for running the curriculum learning process, by querying the scheduler for the
        data loaders and then running the trainer and evaluator for each curriculum step.

        If logging is enabled, logging is done before the curriculum learning process starts, after each curriculum
        step and after the curriculum learning process ends.

        If baseline training is used, the model is reset to the initial state after each curriculum step.
        """

        self.initialize(
            model=self.model,
            logging_path=self.logging_path,
            logging_dict=self.logging_dict,
        )

        while self.scheduler.has_next():
            # Update scheduler
            self.scheduler.next()

            # Get data loader and parameters for current curriculum step
            self.latest_tdata_loader = self.scheduler.get_train_data_loader()
            self.latest_vdata_loader = self.scheduler.get_validation_data_loader()
            self.latest_edata_loader = self.scheduler.get_test_data_loader()

            # Preprocessing
            self.curriculum_step_preprocessing(
                model=self.model,
                logging_path=self.logging_path,
                logging_dict=self.logging_dict,
            )

            # Start training
            self.trainer.run(**self.kwargs)

            # Evaluate model
            self.evaluator.run(**self.kwargs)

            # Postprocessing
            self.curriculum_step_postprocessing(
                model=self.model,
                logging_path=self.logging_path,
                logging_dict=self.logging_dict,
            )

        self.finalize(
            model=self.model,
            logging_path=self.logging_path,
            logging_dict=self.logging_dict,
        )
