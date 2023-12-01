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
    """

    def __init__(self, start: int, end: int, step: int) -> None:
        super().__init__()

        # Curriculum parameters
        self.curriculum_step = -step + start

        self.start: int = start
        self.step: int = step
        self.end: int = end

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

    def get_parameters(self, overview: bool = False) -> dict:
        raise NotImplementedError("get_parameters method is not implemented")
