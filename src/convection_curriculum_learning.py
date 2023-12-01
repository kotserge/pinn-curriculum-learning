import numpy as np
import time
import os
from tqdm import tqdm

import torch
from torch import nn, optim, Tensor
from torch.utils import data
from torch.nn.modules.loss import _Loss

from data.pde import PDESolver, ConvectionEquationSolver
from data.augmentation import add_noise

from curriculum_learning.curriculum_scheduler import CurriculumScheduler
from curriculum_learning.curriculum_trainer import CurriculumTrainer
from curriculum_learning.curriculum_evaluator import CurriculumEvaluator
from curriculum_learning.curriculum_learning import CurriculumLearning

from model.model import PINNModel

from util.training_loop import TrainingFactory


# --- Loss ---
# This is the loss function for the convection equation PDE.


def convection_loss(
    input: Tensor,
    target: Tensor,
    x: Tensor,
    t: Tensor,
    convection: float,
    model: nn.Module,
) -> Tensor:
    loss_mse = torch.nn.MSELoss()(input, target)

    loss_pde = ConvectionEquationSolver.loss(
        x=x,
        t=t,
        c=convection,
        model=model,
    )
    loss_pde = torch.mean(loss_pde**2)  # PDE loss

    return loss_mse + loss_pde


# --- PDE Dataset---
# This is the dataset for the convection equation PDE.


class ConvectionEquationPDEDataset(data.Dataset):
    def __init__(
        self,
        l: float = 2 * np.pi,
        t: int = 1,
        n: int = 50,
        convection: int = 1,
        snr: int = 0,
    ):
        """Builds a dataset for the convection equation PDE.

        Args:
            l (float): length of the domain
            t (int): time of the simulation
            n (int): number of grid points
            convection (int): convection coefficient
        """
        super().__init__()
        self.pde = ConvectionEquationSolver(L=l, T=t, N=n, Convection=convection)

        # Generate data
        self.pde.solve_analytic()
        self.X, self.Y = self.pde.store_solution()

        if snr > 0:
            self.Y = add_noise(self.Y, snr=snr)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# --- Curriculum Learning ---
# This is the curriculum learning process for the convection equation PDE.
# It only extends the logging functions to log to wandb.


class ConvectiveCurriculumLearning(CurriculumLearning):
    def init_logging(self, **kwargs) -> None:
        """Initial logging, before the curriculum learning process starts."""
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        # create directory for model
        os.makedirs(f"model/{self.timestamp}", exist_ok=True)

    def curriculum_step_logging(self, **kwargs) -> None:
        """Logging for each curriculum step."""
        print(f"Current curriculum step: {self.scheduler.curriculum_step}")
        torch.save(
            self.model.state_dict(),
            f"model/{self.timestamp}/convection_with_curriculum_{self.scheduler.curriculum_step}.pth",
        )

    def end_logging(self, **kwargs) -> None:
        """Logging after the curriculum learning process ends."""
        pass


# --- Curriculum Scheduler ---
# This is the scheduler for the convection equation PDE.


class ConvectionCurriculumScheduler(CurriculumScheduler):
    """Scheduler for curriculum learning for the convection equation PDE.

    Args:
        CurriculumScheduler (CurriculumScheduler): base class for curriculum scheduler
    """

    def __init__(
        self,
        start: int,
        end: int,
        step: int,
        epochs: int,
        batch_size: int,
        l: float = 2 * np.pi,
        t: int = 1,
        n: int = 50,
        snr: int = 0,
    ) -> None:
        """Builds a scheduler for the convection equation PDE.

        Args:
            start (int): starting curriculum step
            end (int): ending curriculum step
            step (int): curriculum step size
            epochs (int): number of epochs
            batch_size (int): batch size
            l (float, optional): length of the domain. Defaults to 2 * np.pi.
            t (int, optional): time of the simulation. Defaults to 1.
            n (int, optional): number of grid points. Defaults to 50.
            snr (int, optional): noise to be added. If zero, no noise will be added to solution. Defaults to 0.
        """
        super().__init__(start, end, step)

        # Parameters for training
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        # PDE parameters
        self.l: float = l
        self.t: int = t
        self.n: int = n
        self.snr: int = snr

    def get_train_data_loader(self, **kwargs) -> data.DataLoader:
        """Returns a data loader for the current curriculum step."""
        return data.DataLoader(
            ConvectionEquationPDEDataset(
                l=self.l,
                t=self.t,
                n=self.n,
                convection=self.curriculum_step,
                snr=self.snr,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            **kwargs,
        )

    def get_validation_data_loader(self, **kwargs) -> data.DataLoader:
        """Returns a data loader for the current curriculum step."""
        return self.get_train_data_loader(**kwargs)

    def get_test_data_loader(self, **kwargs) -> data.DataLoader:
        """Returns a data loader for the current curriculum step."""
        return self.get_train_data_loader(**kwargs)

    def get_parameters(self, overview: bool = False) -> dict:
        """Returns parameters for the current curriculum step."""

        if overview:
            return {
                "curriculum": {
                    "curriculum_step": list(range(self.start, self.end + 1, self.step)),
                    "start": self.start,
                    "step": self.step,
                    "end": self.end,
                },
                "training": {
                    "epochs": self.epochs,
                },
                "pde": {
                    "l": self.l,
                    "t": self.t,
                    "n": self.n,
                    "convection": self.curriculum_step,
                    "snr": self.snr,
                },
            }

        return {
            "curriculum": {
                "curriculum_step": self.curriculum_step,
                "start": self.start,
                "step": self.step,
                "end": self.end,
            },
            "training": {
                "epochs": self.epochs,
            },
            "pde": {
                "l": self.l,
                "t": self.t,
                "n": self.n,
                "convection": self.curriculum_step,
                "snr": self.snr,
            },
        }


# --- Trainer ---
# This is the trainer for the convection equation PDE.


class ConvectionEquationTrainer(CurriculumTrainer):
    def run(self, **kwargs) -> None:
        """Runs the training process."""

        epoch_loss_mean = 0.0

        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        for _ in tqdm(range(self.parameters["training"]["epochs"]), miniters=0):
            # Epoch loss aggregator
            epoch_loss_aggregation = 0.0

            # Batch loop
            for batch, (data_inputs, data_labels) in enumerate(self.train_data_loader):
                ## Step 1 - Move input data to device
                data_inputs, data_labels = data_inputs.to(self.device), data_labels.to(
                    self.device
                ).to(torch.float64)

                # Step 1.5 - Change the data input
                x, t = data_inputs[:, 0].unsqueeze(1), data_inputs[:, 1].unsqueeze(1)

                ## Step 2 - Run the model on the input data
                prediction = self.model(x, t)  # .squeeze(dim=1)

                ## Step 3 - Calculate the loss using the module loss_module
                loss = self.loss_module(
                    prediction,
                    data_labels,
                    x,
                    t,
                    self.parameters["pde"]["convection"],
                    model,
                )

                ## Step 4 - Perform backpropagation & update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Step 5 - Accumulate loss for current epoch
                epoch_loss_aggregation += loss.item()

            # Step 6 - Average epoch loss
            epoch_loss_mean += epoch_loss_aggregation / len(self.train_data_loader)

        # Step 8 - Final logging
        print("-" * 50 + "\nTraining finished\n" + "-" * 50)
        print(f"Epoch loss: {epoch_loss_mean / self.parameters['training']['epochs']}")


# --- Evaluator ---
# This is the evaluator for the convection equation PDE.


class ConvectionEquationEvaluator(CurriculumEvaluator):
    def run(self, **kwargs) -> None:
        pass


if __name__ == "__main__":
    model = PINNModel(input_dim=2, hidden_dim=50).to(torch.float64)
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.01)
    loss = convection_loss

    scheduler = ConvectionCurriculumScheduler(
        start=1,
        end=30,
        step=1,
        epochs=50,
        batch_size=64,
        l=2 * np.pi,
        t=1,
        n=50,
        snr=0,
    )

    learner = ConvectiveCurriculumLearning(
        model,
        optimizer,
        loss,
        scheduler,
        ConvectionEquationTrainer,
        ConvectionEquationEvaluator,
    )
    learner.run()
