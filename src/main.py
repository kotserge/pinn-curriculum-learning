import sys
import yaml

import wandb
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

print("* Initializing curriculum learning components")

# Init curriculum learning components based on hyperparameters
# No model, optimizer or loss module is passed to the curriculum learning components
# as they are initialized in the curriculum learning components themselves
learner = experiment.ConvectiveCurriculumLearning(
    modelzz=None,
    optimizerzz=None,
    losszz=None,
    schedulerzz=experiment.ConvectionCurriculumScheduler,
    trainerzz=experiment.ConvectionEquationTrainer,
    evaluatorzz=experiment.ConvectionEquationEvaluator,
    config=hyperparameters,
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
    wandb.agent(sweep_id, function=learner.run, count=hyperparameters["sweep"]["count"])
    exit()


print("* Starting curriculum learning")
print("-" * 50)
learner.run()
