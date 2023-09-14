# Variable Binding Mechanisms in Recurrent Neural Networks

Tl;dr - check out the demo in `demo_rcopy.ipynb`

The repository contains files for exploring variable binding mechanisms in recurrent architectures. The repo
serves as a companion to the paper [Episodic Memory Theory of Recurrent Neural Networks: Insights into
Long-Term Information Storage and Manipulation](https://openreview.net/pdf?id=PYoEjBFAIM).

## Installation

The tools are written in Python 3.6 and require the following packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pytorch`
- `pytorch-lightning`
- `tensorboard`  (for logging)

to install run the following in the root directory of the repository:

```pip install .```

this will install the package in editable mode.

The variable memory analysis presented in the paper also requires ```emt-tools``` from this 
repository: 
[https://github.com/arjunkaruvally/emt-tools](https://github.com/arjunkaruvally/emt-tools)

## Usage

Prior to use, create ```EXPERIMENT_OUTPUT_DIR``` environment variable with the appropriate
path to the directory where you want to store/read the results of the experiments.

Experiments are in the form of python scripts in the ```experiments``` directory. 
The scripts can be run from the root directory of the repository as follows:

```python experiments/<experiment_dir>/<experiment_name>.py```

The py files are files that are used for training the models on various tasks. 
Currently, these are the tasks that are available:

- repeat copy: ```1_0_repeat_copy.py```

Python notebooks of experiments can also be found in the respective experiment directory in the
```experiment_notebooks``` directory. These notebooks are used for plotting the results of 
interpreting the experiments.

