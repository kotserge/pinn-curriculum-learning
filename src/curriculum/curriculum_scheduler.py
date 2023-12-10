from torch.utils.data import DataLoader


class CurriculumScheduler:
    """Base class for curriculum scheduler.

    This class is responsible for the parameterized data loading used during each curriculum step.
    The scheduling process is defined by the user by extending this class and implementing the
    get_train_data_loader, get_validation_data_loader and get_test_data_loader methods.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """Initializes the curriculum scheduler.

        Args:
            config (dict): Configuration dictionary containing the Configuration for the curriculum scheduler.
        """
        super().__init__()
        assert "start" in config, "start parameter is required"
        assert "end" in config, "end parameter is required"
        assert "step" in config, "step parameter is required"

        # Curriculum parameters
        self.start: int = config["start"]
        self.step: int = config["step"]
        self.end: int = config["end"]
        self.curriculum_step = self.start - self.step

        self.config: dict = config

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
