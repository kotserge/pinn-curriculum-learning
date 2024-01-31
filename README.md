# Physics Informed Neural Networks (PINNs) for Solving Partial Differential Equations (PDEs)

PINNs are a class of neural networks that can be used to solve partial differential equations (PDEs) by introducing physical domain knowledge into the network through the loss function. Performance of such approaches are highly dependent on the complexity of the underlying PDE and even for moderately complex PDEs might outright fail. [Krishnapriyan, Aditi, et al. "Characterizing possible failure modes in physics-informed neural networks." Advances in Neural Information Processing Systems 34 (2021): 26548-26560.](https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf) ([Archive Link](https://arxiv.org/abs/2109.01050)) ([Code](https://github.com/a1k12/characterizing-pinns-failure-modes)) propose a curriculum learning approach to improve the performance of PINNs, by gradually increasing the complexity of the PDE, allowing the network to train on simpler examples first and then gradually increasing the complexity. This approach can be seen as a smart way of initializing the networks weights and biases.

This project aims to reproduce the results of the paper as well as investigate the effects of sampling size and noise in the training data on the performance of PINNs. 

## Setup

### Model

The model is implemented in PyTorch. The model is a fully connected neural network with 3 hidden layers and 50 neurons per layer. The activation function is the hyperbolic tangent function. The input of the network is the spatial coordinate $`x`$ and temporal coordinate $`t`$. The output of the network is $` \hat u(x, t)`$.

### PDE

The PDE in the initial experiments is the [Convection–Diffusion Equation](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation) (CDE) with no diffusion term and a scalar convection term with a continuous boundary as well as a sine wave as the initial condition.

### Loss Function

The loss function is the mean squared error (MSE) between the predicted and the actual value and the PDE . The loss function is given by

```math 
    \text{MSE}}(\hat u, u \mid \theta) \\
    \mathcal{L}_{\text{PDE}}(\hat u \mid \theta) &= \frac{\partial \hat u}{\partial t} + c \frac{\partial \hat u}{\partial x}\\
    \mathcal{L}_{\text{MSE}}(\hat u, u \mid \theta) &= \frac{1}{n}\sum^n_{i=1}(\hat{u}(x_i, t_i) - u(x_i, t_i))^2
```

where $`\hat u`$ is the learned primitive function, $`u`$ is the actual function, $`\theta`$ are the learned weights and biases of the network, $`\mathcal{L}_{\text{PDE}}`$ is the loss function for the PDE and $`\mathcal{L}_{\text{MSE}}`$ is the MSE loss function.
Note, the loss function differs from the paper, where only the initial condition, boundary condition and the PDE itself are used in the loss function. However, as in our experiments we sample over the whole domain, the boundary and initial conditions are represented to some degree.

### Optimization

We use three different optimizers; Adam, SGD and LBFGS. The hyperparameters for the optimizers are found using sweeps on some idealized experiments, where we sample 100 points from the domain, the noise is set to 50, and each optimizer is given 250 epochs. The hyperparameters swept include learning rate and weight decay for Adam and SGD including the momentum parameter for SGD. For LBFGS we sweep the learning rate, max iterations and history size. The hyperparameters are swept using [Weights & Biases](https://wandb.ai/site) (WandB). The sweeps are defined in the `config` directory under `config/hyperparameter_search`.

### Experiments

As the initial research question is the effect of sample size and noise in the training data, we run experiments with different sample sizes and noise levels. The experiments are defined in the `config` directory under `config/sampling-noise`. For the sample size we sweep over 10, 50, 100, 500, 1000 and 5000 samples. The samples are randomly sampled from the domain consisting of 10000 equidistant points. For the noise we use SNR values in dB of 0 (no noise), 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50.
For the sweep we use the optimal hyperparameters found in the hyperparameter search and use grid search to test all combinations of sample size and noise level (this is done multiple times to get a better estimate of the performance).

### Results

The results can be found [here](https://wandb.ai/singing-kangaroo/Curriculum%20Learning%20Convection%20Equation/workspace?workspace=user-serge-kotchourko). (Note, this is still a work in progress)

## Implementation

### Curriculum Learning Implementation

![curriculum_learning](doc/img/curriculum_loop.drawio.png)

### Project Structure

The project is structured as follows:

```bash
.
├── LICENSE
├── README.md
├── requirements.txt
├── config          # Configuration files for training
├── data            
│   └── runs        # Artifacts created during learning
├── doc             # Some documentation, presentations, etc.            
├── scripts         # Helper scripts
├── src             # Source code
│   ├── main.py     # Main Script
│   ├── curriculum  # Curriculum Learning Implementation
│   ├── data        # Data Handling and Generation
│   ├── experiments # Experiment Implementations
│   ├── loss        # Loss Implementations
│   ├── models      # Model Implementations
│   └── utils       # Utility Functions
└── tmp             # Temporary Files
```

#### Configuration Files

To track our experiments we use [Weights & Biases](https://wandb.ai/site) (WandB). The configuration files are used to configure the experiments and sweeps. The configuration files are written in YAML and are located in the `config` folder. Under the sub-directory `config/examples` you can find some examples for the configuration files.

#### Data

The data folder contains the artifacts created during the training. The artifacts are stored in the `data/runs` folder. The artifacts are stored in the following structure:

```bash
data/runs
├── <run_id_1>
│   ├── images
│   │   ├── some-image.png
│   │   ...
│   ├── model
│   │   ├── checkpoint-curriculum-step-0.pth
│   │   ├── ...
│   │   ├── checkpoint-curriculum-step-<n>.pth
│   │   └── checkpoint-final.pth
...
```

#### Scripts

The scripts folder contains helper scripts. Currently the following scripts are available:

- `scripts/run-exp.sh <n> <config file/_dir>`: Runs the experiment with the given configuration file or directory `n` times.

#### Documentation

The documentation folder contains some documentation, presentations, etc.

#### Source Code

The source code is located in the `src` folder. The source code is structured as follows:

- `curriculum`: Curriculum Learning Implementation
- `experiments`: Experiment Implementations extending the curriculum learning implementation
- `data`: Data Handling and Generation
- `loss`: Loss Implementations
- `model`: Model Implementations
- `utils`: Utility Functions

### Notes on WandB

#### Sweeps

If you want to configure a sweep, you need to re-define the whole block in the sweep block. This is due to how wandb handles nested dictionaries. 
The following is an example taken from `sweep-mini-test.yaml`, re-defining the training block in the sweep block giving epochs 10 and 20 as values for the sweep, while keeping the stopping block the same (compare to `online-mini-test.yaml`). Also note, the original training needs to be removed.

```yaml
sweep:
  name: test-adam-curriculum-learning-sweep
  description: Test sweep for Adam Optimizer and Curriculum Learning
  method: random
  metric:
    name: Loss Overall
    goal: minimize
  parameters:
    training:
      parameters:
        epochs:
          values: [10, 20]
        stopping:
          parameters:
            patience:
              value: 10
```

Note, currently conditional sweeps are not supported, meaning that we have to either carefully choose the parameters or run multiple sweeps over the dependent parameters. Another approach from this [issue](https://github.com/wandb/wandb/issues/1487).
