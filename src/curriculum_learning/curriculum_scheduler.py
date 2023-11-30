from torch.utils.data import DataLoader


class CurriculumScheduler:
    """Scheduler for curriculum learning.

    This class is responsible for scheduling the curriculum learning process, by providing the
    data loader and parameters for each curriculum step.

    This is the base class for all curriculum schedulers and it should be extended by the user for
    each specific case.

    Args:
        step_size (int): curriculum step size
        max_iter (int): maximum curriculum step
    """

    def __init__(self, step_size: int, max_iter: int) -> None:
        super().__init__()

        # Curriculum parameters
        self.curriculum_step = -step_size
        self.step_size: int = step_size
        self.max_iter: int = max_iter

    def next(self) -> None:
        self.curriculum_step += self.step_size

    def has_next(self) -> bool:
        return self.curriculum_step + self.step_size < self.max_iter

    def get_data_loader(self) -> DataLoader:
        raise NotImplementedError("get_data_loader method is not implemented")

    def get_eval_data_loader(self) -> DataLoader:
        raise NotImplementedError("get_eval_data_loader method is not implemented")

    def get_parameters(self) -> dict:
        raise NotImplementedError("get_parameters method is not implemented")
