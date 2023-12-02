from typing import Tuple, Union

import numpy as np

import torch
from torch import nn, Tensor

from .pde import PDESolver


class ConvectionPDESolver(PDESolver):
    """Solves the convection equation using analytical methods.
    See https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation

    Args:
        PDESolver (class): The base class for PDE solvers.
    """

    def __init__(
        self,
        spatial: float,
        temporal: float,
        grid_points: Union[int, Tuple[int, int]],
        convection: float,
    ):
        """Initializes the convection equation solver.

        Args:
            spatial (float): The spatial domain size [0, L].
            temporal (float): The temporal domain size [0, T].
            grid_points (int | (int, int)): The number of grid points in the spatial and temporal domain.
            convection (float): The convection coefficient.
        """
        super().__init__()

        self.L = spatial  # Spatial domain
        self.T = temporal  # Temporal domain
        self.N = grid_points  # Number of grid points

        self.dx = self.L / self.N if isinstance(self.N, int) else self.L / self.N[0]
        self.dt = self.T / self.N if isinstance(self.N, int) else self.T / self.N[1]

        self.u = None  # Solution array

        self.convection = convection

    def u0(self, x: np.ndarray) -> np.ndarray:
        """Initial condition for the convection equation.

        Args:
            x (np.ndarray): The spatial coordinates.

        Returns:
            np.ndarray: The initial condition at the given spatial coordinates.
        """
        return np.sin(x)

    def solve(self) -> None:
        """Computes the solution of the convection equation using analytical methods.
        The solution size is (N, N), where N is the number of grid points in the spatial domain and the time domain.

        The solution is stored in the class variable u.
        """
        x = np.linspace(0, self.L, self.N)
        u0 = self.u0(x)

        # Compute wavenumbers
        k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)

        # Compute Fourier coefficients of initial condition
        u0_hat = np.fft.fft(u0)

        u = np.zeros((self.N, self.N))
        u[0, :] = u0

        # Compute analytical solution using the method of characteristics
        for i in range(1, self.N):
            u_hat_t = u0_hat * np.exp(-1j * self.convection * k * i * self.dt)
            # Transform the solution back to spatial domain using inverse FFT
            u_t = np.real(np.fft.ifft(u_hat_t))
            u[i, :] = u_t

        self.u = np.transpose(
            u
        )  # to present the u in the form of u(x,t) to  be consistent with NN approximation

    def solution(self) -> (np.ndarray, np.ndarray):
        """Returns the input and solution of the convection equation.


        The input is in the form of a 2D array, where the first dimension represents the time and the second dimension represents the spatial domain.
        The solution is a 1D array, corresponding to the evaluation of the function at the input points.
        The length of the array is equal to the number of grid points in the spatial domain multiplied by the number of grid points in the time domain.

        Returns:
            (np.ndarray, np.ndarray): The input and solution of the convection equation.
        """
        # Prepare the input and output data for the neural network
        input_data = np.zeros((self.N * self.N, 2))
        output_data = np.zeros((self.N * self.N, 1))

        # Populate the input and output data arrays
        index = 0
        for i in range(self.N):
            for j in range(self.N):
                input_data[index] = [j * self.dx, i * self.dt]
                output_data[index] = self.u[j, i]
                index += 1

        return input_data, output_data

    @staticmethod
    def loss(x: Tensor, t: Tensor, c: float, model: nn.Module) -> Tensor:
        """Provides the loss function for the convection equation.

        Args:
            x (Tensor): The spatial coordinates.
            t (Tensor): The time coordinates.
            c (float): The convection coefficient.
            model (nn.Module): The neural network model.

        Returns:
            Tensor: The loss value.
        """
        x.requires_grad = True
        t.requires_grad = True

        u = model(x, t)

        # Compute the predicted derivative
        du_dx = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]

        du_dt = torch.autograd.grad(
            u,
            t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]

        return torch.add(du_dt, torch.mul(c, du_dx))
