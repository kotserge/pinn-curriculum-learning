import os

import time
from tqdm import tqdm
from matplotlib import pyplot as plt

import wandb

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import curriculum
import data
import util

# --- PDE Dataset---


class ConvectionEquationPDEDataset(Dataset):
    """Dataset for the convection equation PDE.

    Args:
        Dataset (class): base class for datasets in PyTorch
    """

    def __init__(
        self,
        spatial: float,
        temporal: float,
        grid_points: int,
        convection: float,
        seed: int,
        snr: float = 0,
    ):
        """Initializes the dataset for the convection equation PDE.

        Args:
            spatial (float): The spatial extent of the PDE.
            temporal (float): The temporal extent of the PDE.
            grid_points (int): The number of grid points in each dimension.
            convection (float): The convection coefficient of the PDE.
            snr (float, optional): The signal-to-noise ratio in dB. Defaults to 0.
        """
        super().__init__()
        self.pde = data.ConvectionPDESolver(
            spatial=spatial,
            temporal=temporal,
            grid_points=grid_points,
            convection=convection,
        )

        # Generate data
        self.pde.solve()
        self.X, self.Y = self.pde.solution()

        if snr > 0:
            self.Y = data.augment_by_noise(self.Y, snr=snr, seed=seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# --- Curriculum Learning ---


class ConvectiveCurriculumLearning(curriculum.CurriculumLearning):
    """Curriculum learning for the convection equation PDE.

    Args:
        CurriculumLearning (CurriculumLearning): base class for curriculum learning
    """

    def initialize(self, **kwargs) -> None:
        """Initial logging, before the curriculum learning process starts.

        Initializes the wandb run and creates a directory for the model and other data.
        """
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        # create wandb run
        wandb.login()

        entity, project, group, job_type, run_mode = self._get_wandb_init_parameters(
            **kwargs
        )

        if self.kwargs["resume_path"]:
            self._resume(group=group, job_type=job_type, run_mode=run_mode, **kwargs)
        else:
            self._setup(group=group, job_type=job_type, run_mode=run_mode, **kwargs)

        _ = wandb.init(
            entity=entity,
            project=project,
            group=group,
            job_type=job_type,
            name=self._name,
            id=self._id,
            mode=run_mode,
            config=self.hyperparameters,
            resume="must" if self.kwargs["resume_path"] else None,
        )

        # Finalize setup, if this is a resumed curriculum learning process skip this
        if not self.kwargs["resume_path"]:
            self._finalize_setup()

    def _get_wandb_init_parameters(self, **kwargs) -> dict:
        """Returns the correct parameters for the wandb init function.

        Returns:
            tuple: Parameters for wandb init function (entity, project, group, job_type, run_mode)
        """
        assert "overview" in self.hyperparameters, "No overview in hyperparameters"
        assert "entity" in self.hyperparameters["overview"], "No entity in overview"
        assert "project" in self.hyperparameters["overview"], "No project in overview"

        entity = self.hyperparameters["overview"]["entity"]
        project = self.hyperparameters["overview"]["project"]

        group = (
            self.hyperparameters["overview"]["group"]
            if "group" in self.hyperparameters["overview"]
            else None
        )

        job_type = (
            self.hyperparameters["overview"]["job_type"]
            if "job_type" in self.hyperparameters["overview"]
            else None
        )

        run_mode = (
            self.hyperparameters["overview"]["run_mode"]
            if "run_mode" in self.hyperparameters["overview"]
            and self.hyperparameters["overview"]["run_mode"]
            else "disabled"
        )

        return entity, project, group, job_type, run_mode

    def _setup(self, **kwargs) -> None:
        """Helper function for setup, if this a new curriculum learning process."""
        # Model and Optimizer initialization
        self._id = wandb.util.generate_id()
        self._name = self.hyperparameters["overview"]["experiment"] + "-" + self._id

        # Table logging
        self.logging_dict["Epoch Loss"] = wandb.Table(
            columns=["Curriculum Step", "Epoch", "Loss"]
        )
        self.logging_dict["Early Stopping Hit"] = wandb.Table(
            columns=["Curriculum Step", "Relative Epochs Ratio"]
        )

    def _resume(self, **kwargs) -> None:
        """Helper function for setup, if this is a resumed curriculum learning process."""
        state_dict = torch.load(self.kwargs["resume_path"], map_location=self.device)

        # Model and Optimizer state restoration
        torch.manual_seed(state_dict["Seed"])
        self.hyperparameters["learning"]["seed"] = state_dict["Seed"]
        self.model.load_state_dict(state_dict["Model State Dict"])
        self.optimizer.load_state_dict(state_dict["Optimizer State Dict"])

        # Restore curriculum state
        self._id = state_dict["ID"]
        self._name = state_dict["Name"]
        self.scheduler.curriculum_step = state_dict["Curriculum Step"]
        self.logging_dict = state_dict["Logging Dict"]

    def _finalize_setup(self, **kwargs) -> None:
        """Helper function for setup, if this is a new curriculum learning process."""
        # create directory for model
        self.logging_path = f"data/run/{self.timestamp}-{self._name}"
        self.model_path = f"{self.logging_path}/model"
        self.image_path = f"{self.logging_path}/images"

        os.makedirs(self.logging_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)

        # Initialize scheduler, model, optimizer
        self.scheduler = self.schedulerzz(wandb.config)

        self.model = util.initializer.initialize_model(wandb.config["model"])
        self.model.to(torch.float64).to(self.device)

        self.optimizer = util.initializer.initialize_optimizer(
            wandb.config["optimizer"], self.model
        )

    def curriculum_step_preprocessing(self, **kwargs) -> None:
        """initialization of the loss module."""
        # Initialize loss module
        self.loss = util.initializer.initialize_loss(
            wandb.config["loss"],
            curriculum_step=self.scheduler.curriculum_step,
            model=self.model,
        )

        # Initialize trainer and evaluator, they need the loss module to be initialized
        super().curriculum_step_preprocessing()

    def curriculum_step_postprocessing(self, **kwargs) -> None:
        """Logging for each curriculum step.

        Saves the model after each curriculum step.
        """
        # Save intermediate model
        torch.save(
            {
                "ID": self._id,
                "Name": self._name,
                "Seed": wandb.config["learning"]["seed"],
                "Model State Dict": self.model.state_dict(),
                "Optimizer State Dict": self.optimizer.state_dict(),
                "Curriculum Step": self.scheduler.curriculum_step,
                "Logging Dict": self.logging_dict,
            },
            f"{self.model_path}/model_curriculum_step_{self.scheduler.curriculum_step}.tar",
        )

        # Commit logged data during curriculum step
        wandb.log(data={}, commit=True, step=self.scheduler.curriculum_step)

        # Call super class method, for baseline training resetting the model to initial state
        super().curriculum_step_postprocessing()

    def finalize(self, **kwargs) -> None:
        """Logging after the curriculum learning process has finished.

        Saves the final model and finishes the wandb run.
        """

        # Save final model
        torch.save(
            {
                "ID": self._id,
                "Name": self._name,
                "Seed": wandb.config["learning"]["seed"],
                "Model State Dict": self.model.state_dict(),
                "Optimizer State Dict": self.optimizer.state_dict(),
                "Curriculum Step": self.scheduler.curriculum_step,
                "Logging Dict": self.logging_dict,
            },
            f"{self.model_path}/model_final.tar",
        )

        # log final logging dict
        wandb.log(
            {
                "Epoch Loss": self.logging_dict["Epoch Loss"],
                "Epoch Loss Plot": self._visualize_epoch_loss(),
                "Early Stopping Hit": self.logging_dict["Early Stopping Hit"],
            },
            commit=True,
        )

        wandb.finish()

    def _visualize_epoch_loss(self) -> go.Figure:
        """Visualizes the epoch loss.

        Returns:
            go.Figure: The figure
        """
        df = self.logging_dict["Epoch Loss"].get_dataframe()
        return px.line_3d(
            df,
            x="Epoch",
            y="Curriculum Step",
            z="Loss",
            line_group="Curriculum Step",
            labels={
                "Curriculum Step": "Curriculum Step",
                "Epoch": "Epoch",
                "Loss": "Loss",
            },
        )


# --- Curriculum Scheduler ---


class ConvectionCurriculumScheduler(curriculum.CurriculumScheduler):
    """Scheduler for curriculum learning for the convection equation PDE.

    Args:
        CurriculumScheduler (class): base class for curriculum scheduler
    """

    def __init__(
        self,
        config: dict,
    ) -> None:
        """Initializes the scheduler for curriculum learning for the convection equation PDE.

        Args:
            config (dict): Configuration of the curriculum learning process.
        """
        super().__init__(config["scheduler"])

    def get_train_data_loader(self, **kwargs) -> DataLoader:
        """Returns the parameterized train dataset for the PDE of the current curriculum step.

        Returns:
            DataLoader: Parameterized dataset
        """
        return self._get_parameterized_dataloader("train", **kwargs)

    def get_validation_data_loader(self, **kwargs) -> DataLoader:
        """Returns the parameterized validation dataset for the PDE of the current curriculum step.

        Returns:
            DataLoader: Parameterized dataset
        """
        return self._get_parameterized_dataloader("validation", **kwargs)

    def get_test_data_loader(self, **kwargs) -> DataLoader:
        """Returns the parameterized test dataset for the PDE of the current curriculum step.

        Returns:
            DataLoader: Parameterized dataset
        """
        return self._get_parameterized_dataloader("test", **kwargs)

    def _get_parameterized_dataloader(self, mode: str, **kwargs) -> DataLoader:
        """Returns a parameterized dataset for the current curriculum step.

        Returns:
            Dataset: Parameterized dataset
        """
        convection = wandb.config["scheduler"]["data"][mode]["pde"]["convection"]
        if isinstance(convection, list):
            convection = convection[self.curriculum_step]

        dataset = ConvectionEquationPDEDataset(
            spatial=wandb.config["scheduler"]["data"][mode]["pde"]["l"],
            temporal=wandb.config["scheduler"]["data"][mode]["pde"]["t"],
            grid_points=wandb.config["scheduler"]["data"][mode]["pde"]["n"],
            convection=convection,
            seed=wandb.config["learning"]["seed"],
            snr=wandb.config["scheduler"]["data"][mode]["pde"]["snr"],
        )

        return DataLoader(
            dataset=dataset,
            batch_size=len(dataset)
            if wandb.config["scheduler"]["data"][mode]["batch_size"] == "full"
            else wandb.config["scheduler"]["data"][mode]["batch_size"],
            shuffle=wandb.config["scheduler"]["data"][mode]["shuffle"],
            **kwargs,
        )


# --- Trainer ---
# This is the trainer for the convection equation PDE.


class ConvectionEquationTrainer(curriculum.CurriculumTrainer):
    """Trainer for the convection equation PDE.

    Args:
        CurriculumTrainer (class): base class for curriculum trainer
    """

    def stopping_condition(self) -> bool:
        """Checks if the stopping condition is met.

        Returns:
            bool: True if the stopping condition is met, False otherwise.
        """
        # Initialize best loss and counter for early stopping if not already done
        if not hasattr(self, "best_loss"):
            self.best_loss = np.inf
            self.counter = 0

        # Check if loss is better than best loss
        if self._batch_loss.item() < self.best_loss:
            self.best_loss = self._batch_loss.item()
            self.counter = -1

        self.counter += 1

        return self.counter > wandb.config["training"]["stopping"]["patience"]

    def closure(self) -> torch.Tensor:
        """Closure for the optimizer using the MSE and PDE loss.

        Returns:
            torch.Tensor: The loss
        """
        self.optimizer.zero_grad()
        prediction = self.model(self.closure_x, self.closure_t)
        loss, _, _ = self.loss(
            input=prediction,
            target=self.closure_y,
            x=self.closure_x,
            t=self.closure_t,
        )
        loss.backward()
        return loss

    def run(self, **kwargs) -> None:
        """Runs a basic training process."""

        # Set model to training mode
        self.model.train()
        self.optimizer.zero_grad()

        # Check, if stopping condition should be evaluated
        eval_stopping_condition = "stopping" in wandb.config["training"]

        # Epoch loop
        for epoch in tqdm(range(wandb.config["training"]["epochs"]), miniters=0):
            # Epoch loss aggregator
            epoch_loss_aggregation = 0.0

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

                # Step 4 - Accumulate loss for batches in current epoch
                epoch_loss_aggregation += self._batch_loss.item()

            # Step 5 - Epoch logging
            self.logging_dict["Epoch Loss"].add_data(
                self.curriculum_step, epoch, epoch_loss_aggregation
            )

            if (
                eval_stopping_condition and self.stopping_condition()
            ) or self._batch_loss.item() > 1e10:
                break

        # Step 6 - Early stopping logging
        self.logging_dict["Early Stopping Hit"].add_data(
            self.curriculum_step,
            (epoch + 1) / wandb.config["training"]["epochs"],
        )


# --- Evaluator ---
# This is the evaluator for the convection equation PDE.


class ConvectionEquationEvaluator(curriculum.CurriculumEvaluator):
    """Evaluator for the convection equation PDE.

    Args:
        CurriculumEvaluator (class): base class for curriculum evaluator
    """

    def run(self, **kwargs) -> None:
        """Runs the evaluation process.

        Evaluates the model on the test data and logs the results to wandb.
        Evaluations used: MSE, PDE, Overall Loss
        Figure: Comparison of the ground truth and the prediction of the model
        """

        # Set model to evaluation mode
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
                input=prediction,
                target=data_labels,
                x=x,
                t=t,
            )

            # Step 4 - Accumulate loss for current batch
            loss += loss.item()
            loss_mse += loss_mse.item()
            loss_pde += loss_pde.item()

            # Step 5 - Store predictions and ground truth
            predictions.append(prediction)
            ground_truths.append(data_labels)

        # Step 6 - Visualize predictions and ground truth
        fig, _ = util.visualize.comparison_plot(
            prediction=torch.cat(predictions).detach().cpu(),
            ground_truth=torch.cat(ground_truths).detach().cpu(),
            params={
                "title": {
                    "ground_truth": "Analytical PDE Solution",
                    "prediction": "Neural Network PDE Solution",
                },
                "data": {
                    "grid": wandb.config["scheduler"]["data"]["test"]["pde"]["n"],
                    "extent": [
                        0,
                        wandb.config["scheduler"]["data"]["test"]["pde"]["t"],
                        0,
                        wandb.config["scheduler"]["data"]["test"]["pde"]["l"],
                    ],
                },
                "savefig_path": f"{self.logging_path}/images/results_convection_curriculum_{self.curriculum_step}.png",
            },
        )

        # Step 7 - Log to wandb
        wandb.log(
            {
                "Loss Overall": loss,
                "Loss MSE": loss_mse,
                "Loss PDE": loss_pde,
                "Convection Coefficient": wandb.config["scheduler"]["data"]["test"][
                    "pde"
                ]["convection"][self.curriculum_step],
                "PDE Prediction": fig,
            },
            step=self.curriculum_step,
            commit=False,
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
