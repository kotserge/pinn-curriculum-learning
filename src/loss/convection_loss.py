from typing import Any
import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss

import data


class ConvectionMSEPDELoss(_Loss):
    """Provides the MSE + PDE loss for the convection equation.

    Args:
        _Loss (class): The base class for all loss modules in PyTorch
    """

    def __init__(
        self, convection: float, model: nn.Module, regularization: float = 1
    ) -> None:
        """Initializes the loss module.

        Args:
            convection (float): The convection coefficient of the PDE
            model (nn.Module): The model that is used to calculate the PDE loss
            regularization (float, optional): The regularization factor. Defaults to 1.
        """
        super(ConvectionMSEPDELoss, self).__init__()
        self.convection: float = convection
        self.model: nn.Module = model
        self.regularization: float = regularization

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        """Calculates the loss for the convection equation PDE.

        Returns:
            Tensor: The loss
        """
        assert "input" in kwds, "input parameter is required for ConvectionMSEPDELoss"
        assert "target" in kwds, "target parameter is required for ConvectionMSEPDELoss"
        assert "x" in kwds, "x parameter is required for ConvectionMSEPDELoss"
        assert "t" in kwds, "t parameter is required for ConvectionMSEPDELoss"
        return self._loss(
            input=kwds["input"],
            target=kwds["target"],
            x=kwds["x"],
            t=kwds["t"],
        )

    def _loss(
        self,
        input: Tensor,
        target: Tensor,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Calculates the loss for the convection equation PDE.
        The loss is the sum of the MSE loss and the PDE loss.

        Args:
            input (Tensor): Prediction of the model
            target (Tensor): Ground truth
            x (Tensor): Spatial input (on which the prediction is calculated) for the PDE loss (analytical solution)
            t (Tensor): Temporal input (on which the prediction is calculated) for the PDE loss (analytical solution)

        Returns:
            Tensor: The loss
        """
        loss_mse = torch.nn.MSELoss()(input, target)

        loss_pde = data.ConvectionPDESolver.loss(
            x=x,
            t=t,
            c=self.convection,
            model=self.model,
        )
        loss_pde = torch.mean(torch.pow(loss_pde, 2))  # PDE loss

        return (
            torch.add(loss_mse, torch.mul(self.regularization, loss_pde)),
            loss_mse,
            loss_pde,
        )
