import numpy as np

from torch import Tensor


class PDESolver:
    """Base class for PDE solvers.

    This class is responsible for solving the PDE using analytical methods.
    The solution is computed using the solve method, which is expected to be implemented by the user.
    The loss function is also expected to be implemented by the user.
    """

    def __init__(self):
        """Initializes the PDE solver."""
        pass

    def u0(self, x: np.ndarray) -> np.ndarray:
        """Initial condition for the PDE.

        Args:
            x (np.ndarray): The spatial coordinates.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            np.ndarray: The initial condition at the given spatial coordinates.
        """
        raise NotImplementedError("Subclasses must implement the 'solve' method.")

    def solve(self) -> None:
        """Computes the solution of the PDE using analytical methods.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("Subclasses must implement the 'solve' method.")

    def solution(self):
        """Returns the input and solution of the PDE.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            np.ndarray: The solution of the PDE.
        """
        raise NotImplementedError("Subclasses must implement the 'solution' method.")

    @staticmethod
    def loss(predicted: Tensor, ground_truth: Tensor, **kwargs) -> Tensor:
        """Provides the loss function for the PDE.

        Args:
            predicted (Tensor): The predicted values.
            ground_truth (Tensor): The ground truth values.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            Tensor: The loss value.
        """
        raise NotImplementedError("Subclasses must implement the 'loss' method.")
