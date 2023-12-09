import sys
import yaml

import numpy as np

import torch
from torch import optim

import model
import loss
from convection_curriculum_learning import (
    ConvectiveCurriculumLearning,
    ConvectionCurriculumScheduler,
    ConvectionEquationTrainer,
    ConvectionEquationEvaluator,
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
    f"* Project: {hyperparameters['overview']['entity']}/{hyperparameters['overview']['project']};\n"
    f"* Group: {hyperparameters['overview']['group']}\n"
    f"* Experiment: {hyperparameters['overview']['experiment']}"
)

print("* Configuring based on hyperparameters")

# Seeding
seed: int = (
    hyperparameters["learning"]
    if "seed" in hyperparameters["learning"]
    else torch.seed()
)

print(f"* Using seed {seed}")
torch.manual_seed(seed)
hyperparameters["learning"]["seed"] = seed

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

loss = loss.convection_mse_pde

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
