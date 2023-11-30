import os
from typing import Callable, Dict, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

from torch.utils.tensorboard import SummaryWriter


class EvaluationFactory:
    def __init__(self) -> None:
        self.data_loader: DataLoader = None
        self.device: str = "cpu"
        self.metrics: Dict[
            str,
            List[
                Callable[[torch.FloatTensor, int | str | float], float],
                Callable[[float, float], float],
            ],
        ] = {}

    def with_data(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def on_device(self, device: str):
        self.device = device

    def with_metrics(
        self,
        metrics: Dict[
            str,
            List[
                Callable[[torch.FloatTensor, int | str | float], float],
                Callable[[float, float], float],
            ],
        ],
    ):
        self.metrics = metrics

    def build(self):
        assert self.data_loader is not None, "Data loader cannot be None"
        assert self.device is not None, "Device cannot be None"
        assert self.metrics is not None, "Metrics cannot be None"

        return self._eval_model

    def _eval_model(self, model: nn.Module):
        # Initialize metrics
        metrics = {}
        for key, _ in self.metrics.items():
            metrics[key] = 0.0

        model.eval()  # Set model to eval mode

        with torch.no_grad():  # Deactivate gradients for the following code
            for data_inputs, data_labels in self.data_loader:
                # Step 1 - Move data to device
                data_inputs, data_labels = data_inputs.to(self.device), data_labels.to(
                    self.device
                )

                # Step 2 - Run the model on the input data
                predictions = model(data_inputs)  # .squeeze(dim=1)

                # Step 3 - Compute metrics
                for key, (metric, _) in self.metrics.items():
                    metrics[key] += metric(predictions, data_labels)

        # Step 4 - Compute final metrics
        for key, (_, final_metric) in self.metrics.items():
            metrics[key] = final_metric(metrics[key], len(self.data_loader.dataset))

        # Step 5 - Report metrics
        self._report_metrics(metrics)

    def _report_metrics(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
