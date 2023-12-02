import time
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

import wandb
import yaml

import numpy as np
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

from util.visualize import comparison_plot

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

    return loss_mse + loss_pde, loss_mse, loss_pde


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

        # create wandb run
        wandb.login()

        group = (
            self.hyperparameters["overview"]["group"]
            if "group" in self.hyperparameters["overview"]
            else None
        )
        run = wandb.init(
            project=self.hyperparameters["overview"]["project"],
            group=group,
            config=self.hyperparameters,
        )
        self._id = run.name

        # create directory for model
        self.run_path = f"data/run/{self._id}-{self.timestamp}/"
        self.model_path = f"{self.run_path}/model/"

        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    def curriculum_step_logging(self, **kwargs) -> None:
        """Logging for each curriculum step."""
        torch.save(
            self.model.state_dict(),
            f"{self.model_path}/model_curriculum_step_{self.scheduler.curriculum_step}.pth",
        )

    def end_logging(self, **kwargs) -> None:
        """Logging after the curriculum learning process ends."""
        torch.save(
            self.model.state_dict(),
            f"{self.model_path}/model_final.pth",
        )
        wandb.finish()


# --- Curriculum Scheduler ---
# This is the scheduler for the convection equation PDE.


class ConvectionCurriculumScheduler(CurriculumScheduler):
    """Scheduler for curriculum learning for the convection equation PDE.

    Args:
        CurriculumScheduler (CurriculumScheduler): base class for curriculum scheduler
    """

    def __init__(
        self,
        hyperparameters: dict,
    ) -> None:
        """Builds a scheduler for the convection equation PDE.

        Hyperparameters args:
            scheduler.curriculum.start (int): starting curriculum step
            scheduler.curriculum.end (int): ending curriculum step
            scheduler.curriculum.step (int): curriculum step size
            scheduler.training.epochs (int): number of epochs
            batch_size (int): batch size
            scheduler.
            t (int): time of the simulation. Defaults to 1.
            n (int): number of grid points. Defaults to 50.
            convection (list[int]): convection coefficient. Defaults to 1.
            snr (int): noise to be added. If zero, no noise will be added to solution. Defaults to 0.
        """
        super().__init__(hyperparameters)

    def get_train_data_loader(self, **kwargs) -> data.DataLoader:
        """Returns a data loader for the current curriculum step."""
        dataset = self._get_parameterized_dataset(**kwargs)
        return data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset)
            if self.hyperparameters["scheduler"]["data"]["batch_size"] == "full"
            else self.hyperparameters["scheduler"]["data"]["batch_size"],
            shuffle=self.hyperparameters["scheduler"]["data"]["shuffle"],
            **kwargs,
        )

    def get_validation_data_loader(self, **kwargs) -> data.DataLoader:
        """Returns a data loader for the current curriculum step."""
        return self.get_train_data_loader(**kwargs)

    def get_test_data_loader(self, **kwargs) -> data.DataLoader:
        """The test data loader returns the same data as the train data loader"""
        dataset = self._get_parameterized_dataset(**kwargs)
        return data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset)
            if self.hyperparameters["scheduler"]["data"]["batch_size"] == "full"
            else self.hyperparameters["scheduler"]["data"]["batch_size"],
            **kwargs,
        )

    def _get_parameterized_dataset(self, **kwargs) -> data.Dataset:
        """Returns a parameterized dataset for the current curriculum step.

        Returns:
            data.Dataset: parameterized dataset
        """
        convection = self.hyperparameters["scheduler"]["pde"]["convection"]
        if isinstance(convection, list):
            convection = convection[self.curriculum_step]

        return ConvectionEquationPDEDataset(
            l=hyperparameters["scheduler"]["pde"]["l"],
            t=hyperparameters["scheduler"]["pde"]["t"],
            n=hyperparameters["scheduler"]["pde"]["n"],
            convection=convection,
            snr=hyperparameters["scheduler"]["pde"]["snr"],
        )


# --- Trainer ---
# This is the trainer for the convection equation PDE.


class ConvectionEquationTrainer(CurriculumTrainer):
    def stopping_condition(self) -> bool:
        """Returns true if the training should stop."""
        # Initialize best loss and counter for early stopping if not already done
        if not hasattr(self, "best_loss"):
            self.best_loss = np.inf
            self.counter = 0

        # Check if loss is better than best loss
        if self._batch_loss.item() < self.best_loss:
            self.best_loss = self._batch_loss.item()
            self.counter = -1
        self.counter += 1

        return self.counter > self.hyperparameters["training"]["stopping"]["patience"]

    def closure(self) -> torch.Tensor:
        """Closure for the optimizer.
        This method is intended to calculate loss terms, propagate gradients and return the loss value.
        Compare: https://pytorch.org/docs/stable/optim.html
        """
        self.optimizer.zero_grad()
        prediction = self.model(self.closure_x, self.closure_t)
        loss, _, _ = self.loss(
            prediction,
            self.closure_y,
            self.closure_x,
            self.closure_t,
            self.hyperparameters["scheduler"]["pde"]["convection"][
                self.curriculum_step
            ],
            model,
        )
        loss.backward()
        return loss

    def run(self, **kwargs) -> None:
        """Runs the training process."""

        # Set model to training mode
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        # Check, if stopping condition should be evaluated
        eval_stopping_condition = "stopping" in self.hyperparameters["training"]

        for _ in tqdm(range(self.hyperparameters["training"]["epochs"]), miniters=0):
            # Batch loop
            for batch, (data_inputs, data_labels) in enumerate(self.train_data_loader):
                ## Step 1 - Move input data to device
                data_inputs, data_labels = data_inputs.to(self.device).to(
                    torch.float64
                ), data_labels.to(self.device).to(torch.float64)

                # Step 2 - Change the data input and move to closure help variables
                x, t = data_inputs[:, 0].unsqueeze(1), data_inputs[:, 1].unsqueeze(1)
                self.closure_x, self.closure_t, self.closure_y = x, t, data_labels

                ## Step 3 - Optimize the model parameters
                self._batch_loss = self._optimize()

            if eval_stopping_condition and self.stopping_condition():
                break

            # Step 4 - Log to wandb
            # TODO: Find out, how to log the loss for each epoch + curriculum step


# --- Evaluator ---
# This is the evaluator for the convection equation PDE.


class ConvectionEquationEvaluator(CurriculumEvaluator):
    def run(self, **kwargs) -> None:
        """Runs the evaluation process."""

        # Set model to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Initialize evaluation metrics
        loss = 0.0
        loss_mse = 0.0
        loss_pde = 0.0

        # Store predictions and ground truth
        predictions = []
        ground_truths = []

        # Loop over batches
        for _, (data_inputs, data_labels) in enumerate(self.data_loader):
            ## Step 1 - Move input data to device
            data_inputs, data_labels = data_inputs.to(self.device).to(
                torch.float64
            ), data_labels.to(self.device).to(torch.float64)

            # Step 1.5 - Change the data input
            x, t = data_inputs[:, 0].unsqueeze(1), data_inputs[:, 1].unsqueeze(1)

            ## Step 2 - Run the model on the input data
            prediction = self.model(x, t)

            ## Step 3 - Calculate the losses
            loss, loss_mse, loss_pde = self.loss(
                prediction,
                data_labels,
                x,
                t,
                self.hyperparameters["scheduler"]["pde"]["convection"][
                    self.curriculum_step
                ],
                model,
            )

            # Step 4 - Accumulate loss for current batch
            loss += loss.item()
            loss_mse += loss_mse.item()
            loss_pde += loss_pde.item()

            # Step 5 - Store predictions and ground truth
            predictions.append(prediction)
            ground_truths.append(data_labels)

        # Step 6 - Visualize predictions and ground truth
        fig, _ = comparison_plot(
            prediction=torch.cat(predictions).detach().cpu(),
            ground_truth=torch.cat(ground_truths).detach().cpu(),
            params={
                "title": {
                    "ground_truth": "Analytical PDE Solution",
                    "prediction": "Neural Network PDE Solution",
                },
                "data": {
                    "grid": self.hyperparameters["scheduler"]["pde"]["n"],
                    "extent": [
                        0,
                        self.hyperparameters["scheduler"]["pde"]["t"],
                        0,
                        self.hyperparameters["scheduler"]["pde"]["l"],
                    ],
                },
                "savefig_path": f"./tmp/results_convection_curriculum_{self.curriculum_step}.png",
            },
        )

        # Step 7 - Log to wandb
        wandb.log(
            {
                "Loss Overall": loss,
                "Loss MSE": loss_mse,
                "Loss PDE": loss_pde,
                "Curriculum Step": self.curriculum_step,
                "Convection Coefficient": self.hyperparameters["scheduler"]["pde"][
                    "convection"
                ][self.curriculum_step],
                "PDE Prediction": fig,
            }
        )

        # Step 8 - Close figure
        plt.close(fig)

        # Step 9 - CLI logging
        print(
            "-" * 50
            + f"\nEvaluation Results for Curriculum Step {self.curriculum_step}\n"
            + "-" * 50
        )
        print(f"Loss: {loss}, MSE: {loss_mse}, PDE: {loss_pde}")
        print("-" * 50)


if __name__ == "__main__":
    # Load hyperparameters
    with open(
        "./config/convection_curriculum_learning_curriculum_Adam.yml", "r"
    ) as file:
        hyperparameters = yaml.safe_load(file)
    # with open("./config/convection_curriculum_learning_test_LBFGS.yml", "r") as file:
    #     hyperparameters = yaml.safe_load(file)

    # Seeding
    if "seed" in hyperparameters["learning"]:
        print(f"Using seed {hyperparameters['learning']['seed']}")
        torch.manual_seed(hyperparameters["learning"]["seed"])
        np.random.seed(hyperparameters["learning"]["seed"])

    # Initialize model, optimizer, loss module and data loader
    model = PINNModel(
        input_dim=hyperparameters["model"]["input_dim"],
        hidden_dim=hyperparameters["model"]["hidden_dim"],
    ).to(torch.float64)

    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters["optimizer"]["lr"],
        weight_decay=hyperparameters["optimizer"]["weight_decay"],
    )
    # optimizer = optim.LBFGS(
    #     model.parameters(),
    #     lr=hyperparameters["optimizer"]["lr"],
    #     history_size=hyperparameters["optimizer"]["history_size"],
    #     max_iter=hyperparameters["optimizer"]["max_iter"],
    # )

    loss = convection_loss

    # Init curriculum learning components
    scheduler = ConvectionCurriculumScheduler(
        hyperparameters=hyperparameters,
    )

    learner = ConvectiveCurriculumLearning(
        model=model,
        optimizer=optimizer,
        loss=loss,
        scheduler=scheduler,
        trainer=ConvectionEquationTrainer,
        evaluator=ConvectionEquationEvaluator,
        hyperparameters=hyperparameters,
    )

    learner.run()
