import sys
import yaml

import wandb

import torch
from torch import optim

import model
import loss
import experiment

resume_path = None
if len(sys.argv) > 1:
    hyperparameters_path = sys.argv[1]

    if len(sys.argv) > 2:
        resume_path = sys.argv[2]

else:
    raise ValueError(
        "Please provide at least a path to the hyperparameters file as a command line argument."
    )

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

# Use dictionary to map model and optimizer names to classes
implemented_models = {
    "ConvectionPINNModel": model.ConvectionPINNModel,
}
implemented_optimizers = {
    "Adam": optim.Adam,
    "LBFGS": optim.LBFGS,
    "SGD": optim.SGD,
}

loss = loss.convection_mse_pde

# Init curriculum learning components
learner = experiment.ConvectiveCurriculumLearning(
    modelzz=implemented_models[hyperparameters["model"]["name"]],
    optimizerzz=implemented_optimizers[hyperparameters["optimizer"]["name"]],
    loss=loss,
    schedulerzz=experiment.ConvectionCurriculumScheduler,
    trainerzz=experiment.ConvectionEquationTrainer,
    evaluatorzz=experiment.ConvectionEquationEvaluator,
    hyperparameters=hyperparameters,
    resume_path=resume_path,
    device=hyperparameters["learning"]["device"],
)

if "sweep" in hyperparameters:
    print("* Starting sweep")
    print("-" * 50)

    sweep_id = wandb.sweep(
        entity=hyperparameters["overview"]["entity"],
        project=hyperparameters["overview"]["project"],
        sweep=hyperparameters["sweep"],
    )
    wandb.agent(sweep_id, function=learner.run)


print("* Starting curriculum learning")
print("-" * 50)
learner.run()
