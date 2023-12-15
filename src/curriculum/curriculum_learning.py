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

    This class is responsible for the curriculum learning process.

    The curriculum learning process is defined by the user by extending this class and implementing the
    initialize, curriculum_step_preprocessing, curriculum_step_postprocessing and finalize methods.

    The initialize method is responsible for initializing the model, optimizer and scheduler.
    The curriculum_step_preprocessing method is responsible for preprocessing before each curriculum step,
        thus initializing the trainer, evaluator and the loss module for the current curriculum step.
    The curriculum_step_postprocessing method is responsible for processing after each curriculum step.
    The finalize method is responsible for finalization after the curriculum learning process ends.

    The curriculum learning process is started by calling the run method.
    """

    def __init__(
        self,
        modelzz: Type[nn.Module],
        optimizerzz: Type[Optimizer],
        losszz: Type[_Loss],
        schedulerzz: Type[CurriculumScheduler],
        trainerzz: Type[CurriculumTrainer],
        evaluatorzz: Type[CurriculumEvaluator],
        config: dict,
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
            losszz (_Loss): The loss module class to be used. This class needs to be initialized with the hyperparameters during the curriculum learning process.
            schedulerzz (CurriculumScheduler): The scheduler class to be used.
            trainerzz (Type[CurriculumTrainer]): The trainer class to be used.
            evaluatorzz (Type[CurriculumEvaluator]): The evaluator class to be used.
            config (dict): Hyperparameters of the whole curriculum learning process.
            logging_path (str, optional): Path to the logging directory. Defaults to None.
            logging_dict (dict, optional): Dictionary containing the logging information. Defaults to None.
            device (str, optional): On which device the process should be run. Defaults to "cuda" if available, otherwise "cpu".
        """
        # Model, optimizer, loss module and data loader
        self.modelzz: nn.Module = modelzz
        self.model: nn.Module = None

        self.optimizerzz: Optimizer = optimizerzz
        self.optimizer: Optimizer = None

        self.losszz: _Loss = losszz
        self.loss: _Loss = None

        # Curriculum learning components
        self.schedulerzz: Type[CurriculumScheduler] = schedulerzz
        self.trainerzz: Type[CurriculumTrainer] = trainerzz
        self.evaluatorzz: Type[CurriculumEvaluator] = evaluatorzz

        # Hyperparameters
        self.hyperparameters: dict = config
        self.baseline: bool = not config["learning"]["curriculum"]

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

        This function is responsible for initializing the model, optimizer and scheduler.
        The model and optimizer should be initialized here, by the user.

        By default the scheduler is  initialized here, and if overriden, the user should
            call super().initialize() or initialize the scheduler manually.
        """

        self.scheduler = self.schedulerzz(
            config=self.config,
            **self.kwargs,
            kwargs=kwargs,
        )

    def curriculum_step_preprocessing(self, **kwargs) -> None:
        """Function for preprocessing before each curriculum step.

        This function is responsible for preprocessing before each curriculum step,
            thus initializing the trainer, evaluator and the loss module for the current curriculum step.
            The loss module should be initialized here, if it is dependent on the curriculum step, else it should be
            initialized in the initialize() function.

        By default the trainer and evaluator are initialized here, and if overriden, the user should
            call super().curriculum_step_preprocessing() or initialize the model and optimizer manually.
        """
        # Get data loader and parameters for current curriculum step
        self.latest_tdata_loader = self.scheduler.get_train_data_loader()
        self.latest_vdata_loader = self.scheduler.get_validation_data_loader()
        self.latest_edata_loader = self.scheduler.get_test_data_loader()

        self.trainer = self.trainerzz(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            train_data_loader=self.latest_tdata_loader,
            validation_data_loader=self.latest_vdata_loader,
            curriculum_step=self.scheduler.curriculum_step,
            config=self.config,
            device=self.device,
            logging_path=self.logging_path,
            logging_dict=self.logging_dict,
            kwargs=kwargs,
        )

        self.evaluator = self.evaluatorzz(
            model=self.model,
            loss=self.loss,
            data_loader=self.latest_edata_loader,
            curriculum_step=self.scheduler.curriculum_step,
            config=self.config,
            device=self.device,
            logging_path=self.logging_path,
            logging_dict=self.logging_dict,
            kwargs=kwargs,
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
