import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvectionPINNModel(nn.Module):
    """Implements the physics-informed neural network model for the convection-diffusion equation.

    Paper: 'Characterizing possible failure modes in physics-informed neural networks.' by Krishnapriyan, Aditi, et al.
    Retrieved from https://arxiv.org/abs/2109.01050

    Args:
        nn (Module): The neural network module from PyTorch.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 50) -> None:
        """Initializes the neural network model.

        Args:
            input_dim (int): Input dimension of the neural network. Defaults to 2.
            hidden_dim (int): Hidden dimension of the neural network. Defaults to 50.
        """
        super(ConvectionPINNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define the neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        x = F.tanh(self.fc1(inputs))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc5(x)
        return x
