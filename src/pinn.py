import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src import Model, PDE, data_augmentation


# choosing the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build the PDE
# @param: L is the domain length, T is the maximum time, N is the number of time and space discretization
# Convection is the pde parameter
pde = PDE.ConvectionEquationSolver(L=2 * np.pi, T=1, N=50, Convection=10)

# Generate training and test data
pde.solve_analytic()
X, Y = pde.store_solution()

# Optional: plot to see how data look like
pde.visualize("./data/ConvectionEquationSolver.png")

# Setup precision
precision = torch.float32


# Split the data into training, evaluation and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.8, random_state=42
)
X_train, X_eval, Y_train, Y_eval = train_test_split(
    X_train, Y_train, test_size=0.5, random_state=42
)


# add noise to training data, SNR: signal to noise ratio
snr = 100
Y_train = data_augmentation.add_noise(Y_train, snr=snr)


# Convert data to torch format and adjsut the precision
X = torch.from_numpy(X).to(precision)
X_train = torch.from_numpy(X_train).to(precision)
X_eval = torch.from_numpy(X_eval).to(precision)
X_test = torch.from_numpy(X_test).to(precision)
Y_train = torch.from_numpy(Y_train).to(precision)
Y_eval = torch.from_numpy(Y_eval).to(precision)
Y_test = torch.from_numpy(Y_test).to(precision)


# Number of training epochs
num_epochs = 30000

# Setup neural network architecture
model = Model.PINNModel(input_dim=2, hidden_dim=50).to(precision)

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.01)

# Loss function
Loss = torch.nn.MSELoss()

for epoch in range(num_epochs):
    # PDE loss
    pde_residual = pde.loss(
        x=X_train[:, 0].unsqueeze(1), t=X_train[:, 1].unsqueeze(1), model=model
    )
    loss_pde = torch.mean(pde_residual**2)  # PDE loss

    # Data residual
    NN_output = model(x=X_train[:, 0].unsqueeze(1), t=X_train[:, 1].unsqueeze(1))
    loss_data = Loss(NN_output, Y_train)

    # Total loss
    loss_total = loss_data + loss_pde

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_total.item()}")
