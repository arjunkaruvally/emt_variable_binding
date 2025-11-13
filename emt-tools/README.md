# emt-tools

This repository will contain a non-exhaustive collection of tools to mechanistically interpret neural networks that 
are derived from the Episodic Memory Theory [Episodic Memory Theory of Recurrent Neural Networks: Insights into
Long-Term Information Storage and Manipulation](https://openreview.net/pdf?id=PYoEjBFAIM).

## Installation

The tools are written in Python 3.6 and require the following packages:
- numpy
- scipy
- matplotlib

to install run the following in the root directory of the repository:

```pip install .```

## Usage

Interpretation of recurrent neural networks are enabled by converting the non-linear dynamics
to linear dynamics. This is done by linearizing the non-linear dynamics around a fixed point.
Whatever method is used to linearize the dynamics, the linear model needs to be converted to
```emt_tools.models.LinearModel``` class. 

This class is a wrapper around the linear model that provides some useful methods for
interpreting the linear model without going too deep into the algorithms used for 
interpretation.

Further, ```emt_tools.utils``` contains many functions that directly implements the interpretation 
algorithms.

For sample usage of ```emt_tools```, check out demo notebooks in [https://github.com/arjunkaruvally/variable_binding_episodic](https://github.com/arjunkaruvally/variable_binding_episodic)
