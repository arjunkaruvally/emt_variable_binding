# Variable Binding Mechanisms in Recurrent Neural Networks

Tl;dr - check out the demo in `demo_rcopy.ipynb` and the collab notebook [https://colab.research.google.com/drive/1msSMZNUlHdF2RVz1NPb2HJ3UIOfwmnPs](https://colab.research.google.com/drive/1msSMZNUlHdF2RVz1NPb2HJ3UIOfwmnPs)

The repository contains files for exploring variable binding mechanisms in recurrent architectures. The repo
serves as a companion to the paper [Episodic Memory Theory of Recurrent Neural Networks: Insights into
Long-Term Information Storage and Manipulation](https://openreview.net/pdf?id=PYoEjBFAIM).

## Installation

The full software is written in Python 3.6 and require the following packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pytorch`
- `pytorch-lightning`
- `tensorboard`  (for logging)

to install run the following in the root directory of the repository:

```pip install .```

The variable memory analysis presented in the paper also requires ```emt-tools``` from this 
repository: 
[https://github.com/arjunkaruvally/emt-tools](https://github.com/arjunkaruvally/emt-tools)

## Usage

Prior to use, create ```EXPERIMENT_OUTPUT_DIR``` environment variable with the appropriate
path to the directory where you want to store/read the results of the experiments.

Experiments are in the form of python scripts in the ```experiments_public``` directory. 
The scripts can be run from the root directory of the repository as follows:

```python experiments/<experiment_dir>/<experiment_name>.py```

The py files are files that are used for training the models on various tasks. 
Currently, these are the tasks that are available:

- repeat copy: ```1_0_repeat_copy.py```
- all_linearVB: ```1_0_binary_linearVB.py``` (also contains a slurm cluster script to train multiple models in cluster)

Python notebooks of experiments can also be found in the respective experiment directory in the
```experiment_notebooks``` directory. These notebooks will typically have interpretability experiments 
on all the variable binding tasks.

## Training and Evaluating Multiple Models

### Unified Configuration System

The repository includes `train_multiple.py` and `evaluate_multiple.py` scripts for training and evaluating multiple models with a unified configuration system. Configurations are automatically saved to the experiment directory, making results fully traceable.

### Training Multiple Models

Train multiple models with configurable parameters:

```bash
# Basic usage with required parameters
python train_multiple.py --experiment_name my_experiment --seq_length 8 --hidden_dim 128

# Train with variable sequence lengths (3 to 7)
python train_multiple.py --experiment_name var_length_exp --seq_length 3 7 --hidden_dim 96

# Customize training parameters
python train_multiple.py \
    --experiment_name custom_exp \
    --seq_length 8 \
    --hidden_dim 128 \
    --num_models 100 \
    --epochs 30 \
    --learning_rate 0.0005 \
    --num_workers 4 \
    --accelerator gpu

# Disable curriculum learning
python train_multiple.py --experiment_name no_curriculum --seq_length 8 --hidden_dim 128 --no_curriculum
```

#### Key Training Arguments:
- `--experiment_name`: Name of experiment (creates a directory)
- `--seq_length`: Sequence length (single value or two values for range)
- `--hidden_dim`: RNN hidden dimension (default: 128)
- `--num_models`: Number of models to train (default: 50)
- `--epochs`: Training epochs (default: 20)
- `--learning_rate`: Learning rate (default: 0.001)
- `--num_workers`: Parallel training processes (default: 2)
- `--accelerator`: Device (auto/gpu/cpu, default: auto)
- `--curriculum`: Enable curriculum learning (default: True)
- `--curriculum_threshold`: Curriculum threshold (default: 0.96)

Run `python train_multiple.py --help` for all options.

### Evaluating Models

Evaluate trained models using the saved configuration:

```bash
# Load configuration from training (recommended)
python evaluate_multiple.py --models_dir my_experiment

# Override sequence lengths
python evaluate_multiple.py --models_dir my_experiment --seq_length 8

# Evaluate on CPU
python evaluate_multiple.py --models_dir my_experiment --accelerator cpu

# Set magnitude threshold for eigenvalue plots
python evaluate_multiple.py --models_dir my_experiment --magnitude_threshold 0.1
```

#### Key Evaluation Arguments:
- `--models_dir`: Directory containing trained models (with config.json)
- `--seq_length`: Override sequence lengths from training config (optional)
- `--test_horizon`: Evaluation timesteps (default: 200)
- `--accelerator`: Device (auto/gpu/cpu, default: auto)
- `--magnitude_threshold`: Min magnitude for eigenvalue plots (default: 0.0)

Run `python evaluate_multiple.py --help` for all options.

## Analyzing Results

After evaluation, use the new **modular analysis framework** for comprehensive analysis:

```bash
# Quick PCA visualization
python analyze.py embedding --summary_file my_experiment/evaluation_summary.json --method pca

# Persistent homology analysis (requires: pip install ripser persim)
python analyze.py persistence --summary_file my_experiment/evaluation_summary.json

# Compare multiple experiments
python analyze.py compare \
    --summary_files exp1/evaluation_summary.json exp2/evaluation_summary.json \
    --labels "Experiment 1" "Experiment 2"
```

**See [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) for complete documentation** including:
- Eigenvalue distribution analysis
- Dimensionality reduction (PCA, t-SNE, MDS)
- Persistent homology and topological data analysis
- Distance metrics and model comparison
- Programmatic API usage

**Key analysis commands:**
- `python analyze.py eigenvalue`: Compute eigenvalue statistics
- `python analyze.py embedding`: PCA/t-SNE/MDS visualizations
- `python analyze.py persistence`: Topological data analysis
- `python analyze.py compare`: Compare multiple experiments

**Migration note:** Old analysis scripts (`pca_signatures.py`, `visualize_wasserstein.py`, etc.) are superseded by the modular framework. See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for migration instructions.

### Configuration Files

Training creates the following structure:
```
experiment_name/
├── config.json              # Training configuration
├── model-1/
│   ├── checkpoint/          # Model checkpoint
│   ├── training.log         # Training log
│   └── lightning_logs/      # TensorBoard logs
├── model-2/
│   └── ...
└── ...
```

Evaluation adds:
```
experiment_name/
├── config.json              # Original training config
├── eval_config.json         # Evaluation configuration
├── evaluation_summary.json  # Evaluation metrics
├── global_analysis/         # Cross-model analysis plots
├── model-1/
│   └── results/             # Per-model evaluation results
├── model-2/
│   └── results/
└── ...
```

### Configuration Format

`config.json` contains all training parameters:
```json
{
  "metadata": {
    "experiment_name": "my_experiment",
    "created_at": "2025-01-15T10:30:00",
    "script": "train_multiple.py"
  },
  "task": {
    "task_id": [255, 0, 0],
    "input_dim": 8,
    "seq_length": 8,
    "use_eos": false
  },
  "model": {
    "type": "rnn",
    "hidden_dim": 128,
    "bias": false
  },
  "training": {
    "seed": 0,
    "learning_rate": 0.001,
    "batch_size": 64,
    "gradient_clip": 1.0,
    "epochs": 20,
    "num_models": 50,
    "curriculum": true,
    "curriculum_threshold": 0.96,
    "curriculum_horizons": [10, 20, 40, 60, 100]
  },
  "execution": {
    "accelerator": "gpu",
    "num_workers": 2
  }
}
```

## Reproducing the results in the paper

The results in the paper can be reproduced by running the following:

- First train the models using the cluster script ```experiments_public/2_all_linearVB/1_0_binary_linearVB.slurm```
- Run the associated notebooks in ```experiments_public/2_all_linearVB/experiment_notebooks/1_0_binary_linearVB.ipynb``` to generate the
  results in the paper.
- If you want to play with training and interpreting individual variable binding tasks, run the ```demo.rcopy.ipynb``` notebook
in the root directory of the repository with various task ids.
