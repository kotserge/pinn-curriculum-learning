import sys
import yaml

import numpy as np

import torch
from torch import optim

import model
from convection_curriculum_learning import (
    ConvectiveCurriculumLearning,
    ConvectionCurriculumScheduler,
    ConvectionEquationTrainer,
    ConvectionEquationEvaluator,
    convection_loss,
)

if len(sys.argv) > 1:
    hyperparameters_path = sys.argv[1]

print("-" * 50, "\nStarting Curriculum Learning\n", "-" * 50)
print(f"* Using hyperparameters file {hyperparameters_path}")

# Load hyperparameters
with open(hyperparameters_path, "r") as file:
    hyperparameters = yaml.safe_load(file)

print("* Hyperparameters loaded")
print(
    f"* Project: {hyperparameters['overview']['project']}; Group: {hyperparameters['overview']['group']}"
)

print("* Configuring based on hyperparameters")
# Seeding
if "seed" in hyperparameters["learning"]:
    print(f"Using seed {hyperparameters['learning']['seed']}")
    torch.manual_seed(hyperparameters["learning"]["seed"])
    np.random.seed(hyperparameters["learning"]["seed"])

# Initialize model, optimizer, loss module and data loader
model = model.PINNModel(
    input_dim=hyperparameters["model"]["input_dim"],
    hidden_dim=hyperparameters["model"]["hidden_dim"],
).to(torch.float64)

if hyperparameters["optimizer"]["name"] == "Adam":
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters["optimizer"]["lr"],
        weight_decay=hyperparameters["optimizer"]["weight_decay"],
    )
elif hyperparameters["optimizer"]["name"] == "LBFGS":
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=hyperparameters["optimizer"]["lr"],
        history_size=hyperparameters["optimizer"]["history_size"],
        max_iter=hyperparameters["optimizer"]["max_iter"],
    )
elif hyperparameters["optimizer"]["name"] == "SGD":
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameters["optimizer"]["lr"],
        momentum=hyperparameters["optimizer"]["momentum"],
        weight_decay=hyperparameters["optimizer"]["weight_decay"],
        nesterov=hyperparameters["optimizer"]["nesterov"],
    )
else:
    raise NotImplementedError(
        f"Optimizer {hyperparameters['optimizer']['name']} not implemented."
    )

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

print("* Starting curriculum learning")
print("-" * 50)
learner.run()
