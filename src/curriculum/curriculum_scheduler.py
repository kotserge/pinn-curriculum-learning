from torch.utils.data import DataLoader


class CurriculumScheduler:
    """Base class for curriculum scheduler.

    This class is responsible for the parameterized data loading used during each curriculum step.
    The scheduling process is defined by the user by extending this class and implementing the
    get_train_data_loader, get_validation_data_loader and get_test_data_loader methods.
    """

    def __init__(self, hyperparameters: dict, **kwargs) -> None:
        """Initializes the curriculum scheduler.

        Args:
            hyperparameters (dict): Hyperparameters of the curriculum learning process. The curriculum parameters are expected to be in the scheduler dictionary.
        """
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

        # Other parameters
        self.kwargs = kwargs

    def next(self) -> None:
        """Proceeds to the next curriculum step."""
        self.curriculum_step += self.step

    def has_next(self) -> bool:
        """Checks if there is a next curriculum step.

        Returns:
            bool: True if there is a next curriculum step, False otherwise.
        """
        return self.curriculum_step + self.step <= self.end

    def reset(self) -> None:
        """Resets the curriculum scheduler to the initial state."""
        self.curriculum_step = self.start

    def get_train_data_loader(self) -> DataLoader:
        """Returns the data loader for the training data for the current curriculum step.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            DataLoader: The data loader for the training data for the current curriculum step.
        """
        raise NotImplementedError("get_train_data_loader method is not implemented")

    def get_validation_data_loader(self) -> DataLoader:
        """Returns the data loader for the validation data for the current curriculum step.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            DataLoader: The data loader for the validation data for the current curriculum step.
        """
        raise NotImplementedError(
            "get_validation_data_loader method is not implemented"
        )

    def get_test_data_loader(self) -> DataLoader:
        """Returns the data loader for the test data for the current curriculum step.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            DataLoader: The data loader for the test data for the current curriculum step.
        """
        raise NotImplementedError("get_test_data_loader method is not implemented")
