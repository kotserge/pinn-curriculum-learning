import numpy as np
import torch
import matplotlib.pyplot as plt


class PDESolver:
    def __init__(self, L, T, N, Convection):
        self.L = L  # Length of the domain
        self.T = T  # Maximum time
        self.N = N  # Number of grid points
        self.dx = L / N  # Grid spacing
        self.dt = T / N  # Time step size

        self.x = np.linspace(0, L, N, endpoint=False)  # Spatial grid
        self.t = np.arange(0, T + self.dt, self.dt)  # Time grid

        self.u = None  # Solution array
        self.X = None  # Vector to store (x, t) coordinates

        self.c = Convection  # convection coefficient

    def u0(self, x):
        raise NotImplementedError("Subclasses must implement the 'solve' method.")

    def solve(self):
        raise NotImplementedError("Subclasses must implement the 'solve' method.")

    def loss(self, X, model):
        raise NotImplementedError("Subclasses must implement the 'solve' method.")

    def store_solution(self):
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

    def visualize(self, path=None):
        # Create meshgrid for plotting

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.imshow(
            self.u,
            cmap="jet",
            aspect="auto",
            extent=[0, self.T, 0, self.L],
            origin="lower",
        )
        plt.colorbar(label="u")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("PDE Solution")
        plt.grid(True)

        if path is not None:
            plt.savefig(path)
        else:
            plt.show()


class ConvectionEquationSolver(PDESolver):
    def u0(self, x):
        return np.sin(x)

    def solve_analytic(self):
        x = np.linspace(0, self.L, self.N)
        u0 = np.sin(x)

        # Compute wavenumbers
        k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)

        # Compute Fourier coefficients of initial condition
        u0_hat = np.fft.fft(u0)

        u = np.zeros((self.N, self.N))
        u[0, :] = u0

        # Compute analytical solution using the method of characteristics
        for i in range(1, self.N):
            u_hat_t = u0_hat * np.exp(-1j * self.c * k * i * self.dt)
            # Transform the solution back to spatial domain using inverse FFT
            u_t = np.real(np.fft.ifft(u_hat_t))
            u[i, :] = u_t

        self.u = np.transpose(
            u
        )  # to present the u in the form of u(x,t) to  be consistent with NN approximation

    def loss(self, x, t, model):
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

        f = du_dt + self.c * du_dx
        return f
