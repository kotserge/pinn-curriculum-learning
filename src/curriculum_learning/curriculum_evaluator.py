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
        loss_module (_Loss): loss module used for training
        data_loader (DataLoader): data loader to use
    """

    def __init__(
        self,
        model: nn.Module,
        loss_module: _Loss,
        data_loader: DataLoader,
        parameters: dict,
        **kwargs
    ):
        self.model: nn.Module = model
        self.loss_module: _Loss = loss_module
        self.data_loader: DataLoader = data_loader
        self.device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.parameters: dict = parameters
        self.kwargs = kwargs

    def run(self) -> None:
        """Runs the evaluation process."""
        raise NotImplementedError("run method is not implemented")
