# Physics Informed Neural Networks (PINNs) for Solving Partial Differential Equations (PDEs)

PINNs are a class of neural networks that can be used to solve partial differential equations (PDEs) by introducing physical domain knowledge into the network through the loss function. Performance of such approaches are highly dependent on the complexity of the underlying PDE and even for moderately complex PDEs might outright fail. [Krishnapriyan, Aditi, et al. "Characterizing possible failure modes in physics-informed neural networks." Advances in Neural Information Processing Systems 34 (2021): 26548-26560.](https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf) ([Archive Link](https://arxiv.org/abs/2109.01050)) ([Code](https://github.com/a1k12/characterizing-pinns-failure-modes)) propose a curriculum learning approach to improve the performance of PINNs, by gradually increasing the complexity of the PDE, allowing the network to train on simpler examples first and then gradually increasing the complexity. This approach can be seen as a smart way of initializing the networks weights and biases.

This project aims to reproduce the results of the paper as well as investigate the effects of sampling size and noise in the training data on the performance of PINNs. 

## Background

For a detailed background on PINNs, curriculum learning and results, please refer to the [project paper](doc/paper.pdf) and the [project presentation](doc/presentation/presentation-final.pdf).

## Implementation

### Curriculum Learning Implementation

![curriculum_learning](doc/img/curriculum_loop.drawio.png)

### Quick Start

#### Requirements

To run the project, you need to have Python 3.8 or higher installed. To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```

#### Running an example experiment

The configuration files are written in YAML and are located in the `config` folder. Under the sub-directory `config/examples` you can find some examples for the configuration files.
The overview section of the configuration file contains configurations for wandb and should be changed to your own wandb project (at least the entity field).

To run an offline example configurations, you can use the following command:

```bash
python src/main.py config/example/offline-mini-test.yml
```

#### Running an experiment

Under the `config` folder, you can find all the configuration files for the experiments used in the paper. To run an experiment, you can use the following command:

```bash
python src/main.py path/to/config.yml
```

Note, that running an experiment will create a new run on wandb, hence the entity field should be changed to your own wandb account or group. If you want to only run the experiment locally, you can set the `run_mode` from `online` to `disabled` in the configuration file. This will disable the wandb logging . Under any run mode, the artifacts will be saved in the `data/runs` folder. This will allow you to see a comparison plot of the different and save intermediate models for each curriculum step. We highly recommend to use wandb, as most data is logged there with additional information.

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
