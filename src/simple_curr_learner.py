import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
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


# Build Dataset Loader for PDE
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


class ConvectionCurriculumScheduler(CurriculumScheduler):
    """Scheduler for curriculum learning for the convection equation PDE.

    Args:
        CurriculumScheduler (CurriculumScheduler): base class for curriculum scheduler
    """

    def __init__(
        self,
        step_size: int,
        max_iter: int,
        epochs: int,
        batch_size: int,
        l: float = 2 * np.pi,
        t: int = 1,
        n: int = 50,
        snr: int = 0,
    ) -> None:
        """Builds a scheduler for the convection equation PDE.

        Args:
            step_size (int): curriculum step size
            max_iter (int): maximum curriculum iteration
            epochs (int): number of epochs
            batch_size (int): batch size
            l (float, optional): length of the domain. Defaults to 2 * np.pi.
            t (int, optional): time of the simulation. Defaults to 1.
            n (int, optional): number of grid points. Defaults to 50.
            snr (int, optional): noise to be added. If zero, no noise will be added to solution. Defaults to 0.
        """
        super().__init__(step_size, max_iter)

        # Parameters for training
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        # PDE parameters
        self.l: float = l
        self.t: int = t
        self.n: int = n
        self.snr: int = snr

    def get_data_loader(self, **kwargs) -> data.DataLoader:
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

    def get_eval_data_loader(self, **kwargs) -> data.DataLoader:
        """Returns a data loader for the current curriculum step."""
        return self.get_data_loader(**kwargs)

    def get_parameters(self) -> dict:
        """Returns parameters for the current curriculum step."""
        return {"epochs": self.epochs, "convection": self.curriculum_step}


class ConvectionEquationTrainer(CurriculumTrainer):
    def run(self, **kwargs) -> None:
        """Runs the training process."""

        epoch_loss_mean = 0.0

        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        for epoch in tqdm(range(self.epochs), miniters=0):
            # Epoch loss aggregator
            epoch_loss_aggregation = 0.0

            # Batch loop
            for batch, (data_inputs, data_labels) in enumerate(self.data_loader):
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
                    model,
                    self.kwargs["convection"],
                )

                ## Step 4 - Perform backpropagation & update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Step 5 - Accumulate loss for current epoch
                epoch_loss_aggregation += loss.item()

            # Step 6 - Average epoch loss
            epoch_loss_mean += epoch_loss_aggregation / len(self.data_loader)

        # Step 8 - Final logging

        # Print training finished
        print("-" * 50 + "\nTraining finished\n" + "-" * 50)
        print(f"Epoch loss: {epoch_loss_mean / self.epochs}")


class ConvectionEquationEvaluator(CurriculumEvaluator):
    def run(self, **kwargs) -> None:
        pass


def convection_loss(output, target, x, t, model, convection):
    loss_mse = torch.nn.MSELoss()(output, target)

    loss_pde = ConvectionEquationSolver.loss(
        x=x,
        t=t,
        c=convection,
        model=model,
    )
    loss_pde = torch.mean(loss_pde**2)  # PDE loss

    return loss_mse + loss_pde


if __name__ == "__main__":
    model = PINNModel(input_dim=2, hidden_dim=50).to(torch.float64)
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.01)

    scheduler = ConvectionCurriculumScheduler(
        step_size=1,
        max_iter=30,
        epochs=50,
        batch_size=64,
        l=2 * np.pi,
        t=1,
        n=50,
        snr=0,
    )

    learner = CurriculumLearning(
        model,
        optimizer,
        convection_loss,
        scheduler,
        ConvectionEquationTrainer,
        ConvectionEquationEvaluator,
    )
    learner.run()
