import torch
from torch import nn
from torch.utils.data import DataLoader


class CurriculumEvaluator:
    """Evaluator class for the curriculum learning process.

    This class is responsible for evaluating the model using the curriculum learning process.
    The training process is defined by the user by extending this class and implementing the
    run method.

    Args:
        model (nn.Module): model to train
        data_loader (DataLoader): data loader to use
    """

    def __init__(self, model: nn.Module, data_loader: DataLoader, **kwargs) -> None:
        self.model: nn.Module = model
        self.data_loader: DataLoader = data_loader
        self.device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.kwargs = kwargs

    def run(self) -> None:
        """Runs the evaluation process."""
        raise NotImplementedError("run method is not implemented")
