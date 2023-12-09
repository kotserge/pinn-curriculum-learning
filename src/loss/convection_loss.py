import torch
from torch import nn, Tensor

import data


def mse_pde(
    input: Tensor,
    target: Tensor,
    x: Tensor,
    t: Tensor,
    convection: float,
    regularization: float,
    model: nn.Module,
) -> Tensor:
    """Calculates the loss for the convection equation PDE.
    The loss is the sum of the MSE loss and the PDE loss.

    Args:
        input (Tensor): Prediction of the model
        target (Tensor): Ground truth
        x (Tensor): Spatial input (on which the prediction is calculated)
        t (Tensor): Temporal input (on which the prediction is calculated)
        convection (float): The convection coefficient of the PDE
        model (nn.Module): The model to be used

    Returns:
        Tensor: The loss
    """
    loss_mse = torch.nn.MSELoss()(input, target)

    loss_pde = data.ConvectionPDESolver.loss(
        x=x,
        t=t,
        c=convection,
        model=model,
    )
    loss_pde = torch.mean(torch.pow(loss_pde, 2))  # PDE loss

    return torch.add(loss_mse, torch.mul(regularization, loss_pde)), loss_mse, loss_pde
