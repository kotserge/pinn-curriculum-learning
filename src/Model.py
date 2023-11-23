import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PINNModel, self).__init__()
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