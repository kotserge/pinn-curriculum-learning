from torch.utils.data import DataLoader


class CurriculumScheduler:
    """Scheduler for curriculum learning.

    This class is responsible for scheduling the curriculum learning process, by providing the
    data loader and parameters for each curriculum step.

    This is the base class for all curriculum schedulers and it should be extended by the user for
    each specific case.

    Args:
        start (int): starting curriculum step
        end (int): ending curriculum step
        step (int): curriculum step size
        hyperparameters (dict): hyperparameters, describing the model, optimizer, loss function
    """

    def __init__(self, hyperparameters: dict) -> None:
        super().__init__()
        assert (
            "start" in hyperparameters["scheduler"]["curriculum"]
        ), "start parameter is required"
        assert (
            "end" in hyperparameters["scheduler"]["curriculum"]
        ), "end parameter is required"
        assert (
            "step" in hyperparameters["scheduler"]["curriculum"]
        ), "step parameter is required"

        # Curriculum parameters
        self.start: int = hyperparameters["scheduler"]["curriculum"]["start"]
        self.step: int = hyperparameters["scheduler"]["curriculum"]["step"]
        self.end: int = hyperparameters["scheduler"]["curriculum"]["end"]
        self.curriculum_step = self.start - self.step

        self.hyperparameters: dict = hyperparameters

    def next(self) -> None:
        self.curriculum_step += self.step

    def has_next(self) -> bool:
        return self.curriculum_step + self.step <= self.end

    def reset(self) -> None:
        self.curriculum_step = self.start

    def get_train_data_loader(self) -> DataLoader:
        raise NotImplementedError("get_train_data_loader method is not implemented")

    def get_validation_data_loader(self) -> DataLoader:
        raise NotImplementedError(
            "get_validation_data_loader method is not implemented"
        )

    def get_test_data_loader(self) -> DataLoader:
        raise NotImplementedError("get_test_data_loader method is not implemented")

    def get_parameters(self) -> dict:
        """Returns the parameters for the current curriculum step."""
        raise NotImplementedError("get_parameters method is not implemented")
