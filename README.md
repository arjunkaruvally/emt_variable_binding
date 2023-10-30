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

## Reproducing the results in the paper

The results in the paper can be reproduced by running the following:

- First train the models using the cluster script ```experiments_public/2_all_linearVB/1_0_binary_linearVB.slurm```
- Run the associated notebooks in ```experiments_public/2_all_linearVB/experiment_notebooks/1_0_binary_linearVB.ipynb``` to generate the 
  results in the paper.
- If you want to play with training and interpreting individual variable binding tasks, run the ```demo.rcopy.ipynb``` notebook 
in the root directory of the repository with various task ids.
