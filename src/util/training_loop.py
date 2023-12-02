import os
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

import tqdm


class TrainingFactory:
    def __init__(self) -> None:
        # Basic training loop
        self.optimizer: Optimizer = None
        self.loss_module: _Loss = None
        self.train_loader: DataLoader = None
        self.num_epochs: int = 50
        self.device: str = "cpu"

        # Also include validation in training loop
        self.validation_loader: DataLoader = None

        # Checkpointing
        self.checkpoint_path: Optional[str] = None
        self.checkpoint_frequency: Optional[int] = None

        # Debugging
        self.verbose: bool = False

    def build(self):
        assert self.optimizer is not None, "Optimizer cannot be None"
        assert self.loss_module is not None, "Loss module cannot be None"
        assert self.train_loader is not None, "Train loader cannot be None"
        assert self.device is not None, "Device cannot be None"

        if self.checkpoint_path is not None:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        return self._train_model

    def with_model(self, model: nn.Module):
        self.model = model
        return self

    def with_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        return self

    def with_loss_module(self, loss_module: _Loss):
        self.loss_module = loss_module
        return self

    def with_training(self, train_loader: DataLoader):
        self.train_loader = train_loader
        return self

    def with_validation(self, validation_loader: DataLoader):
        self.validation_loader = validation_loader
        return self

    def with_epochs(self, num_epochs: int):
        self.num_epochs = num_epochs
        return self

    def on_device(self, device: str):
        self.device = device
        return self

    def with_checkpoints(
        self, checkpoint_path: str = "models/checkpoints", checkpoint_frequency=5
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency
        return self

    def is_verbose(self, verbose: bool = True):
        self.verbose = verbose
        return self

    def _train_model(self, model_name="no_name"):
        # Set time stamp
        self.model_name = model_name

        # Set model to train mode
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        # Training loop
        for epoch in range(self.num_epochs):
            # Epoch loss aggregator
            epoch_loss_aggregation = 0.0

            # Batch loop
            for batch, (data_inputs, data_labels) in enumerate(self.train_loader):
                ## Step 1 - Move input data to device
                data_inputs, data_labels = data_inputs.to(self.device), data_labels.to(
                    self.device
                ).to(torch.float)

                ## Step 2 - Run the model on the input data
                prediction = self.model(data_inputs)  # .squeeze(dim=1)

                ## Step 3 - Calculate the loss using the module loss_module
                loss = self.loss_module(prediction, data_labels)

                ## Step 4 - Perform backpropagation & update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Step 5 - Accumulate loss for current epoch
                epoch_loss_aggregation += loss.item()

                # Step 6 - Batch logging
                self._batch_logging(epoch, batch, loss.item())

            # Step 7 - Epoch logging
            self._epoch_logging(
                epoch, epoch_loss_aggregation, self._validation(self.model)
            )

        # Step 8 - Final logging
        self._final_logging()

    def _validation(self, model: nn.Module):
        if self.validation_loader is None:
            return None

        with torch.no_grad():
            # Validation aggregator
            validation_loss_aggregation = 0.0

            # Validation loop
            for i, (data_inputs, data_labels) in self.validation_loader:
                # Load data to device
                data_inputs, data_labels = data_inputs.to(self.device), data_labels.to(
                    self.device
                ).to(torch.float32)

                # Perform forward pass
                prediction = model(data_inputs).squeeze(dim=1)

                # Calculate loss
                validation_loss_aggregation += self.loss_module(
                    prediction, data_labels
                ).item()

            return validation_loss_aggregation / len(self.validation_loader)

    def _batch_logging(self, epoch: int, batch: int, loss: float):
        # Print batch loss
        if self.verbose:
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}] - Batch [{batch + 1}/{len(self.train_loader)}] - Batch loss: {loss}"
            )

    def _epoch_logging(self, epoch: int, epoch_loss: float, validation_loss: float):
        # Print epoch loss
        if self.verbose:
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}] - Epoch loss: {epoch_loss}"
                + (
                    f" - Validation loss: {validation_loss}"
                    if validation_loss is not None
                    else ""
                )
            )

        # Checkpointing
        if self.checkpoint_path is not None and epoch % self.checkpoint_frequency == 0:
            torch.save(
                self.model.state_dict(),
                f"{self.checkpoint_path}/{self.model_name}_{epoch}.pt",
            )

    def _final_logging(self):
        # Print training finished
        if self.verbose:
            print("-" * 50 + "\nTraining finished\n" + "-" * 50)

        # Checkpointing
        if self.checkpoint_path is not None:
            torch.save(
                self.model.state_dict(),
                f"{self.checkpoint_path}/{self.model_name}_final.pt",
            )
