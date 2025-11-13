import os
import sys
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def simpleBinaryInitialization(seq_length, batch_size, d, p=0.5):
    return torch.bernoulli(p * torch.ones((seq_length, batch_size, d)))
    # return torch.LongTensor(torch.randint(0, 2, size=(seq_length, batch_size, d)))


class SimpleIOPhaseDynamicalSystem(IterableDataset):
    """
    General Framework to sample from a class of dynamical systems characterized by an input and output phases.
    By default the class implements the repeat copy task.
    """

    # NOTE: set horizon as multiple of whatever length atomic operation
    def __init__(
        self,
        composition_op=lambda x, time_id: x[:-1],
        initialization_op=simpleBinaryInitialization,
        bernoulli_p=0.5,
        seed=0,
        d=7,
        batch_size=64,
        seq_length=2,
        horizon=36,
        n_samples=5000,
    ):
        self.d = d
        self.batch_size = batch_size
        self.horizon = horizon
        self.composition_op = composition_op
        self.initialization_op = initialization_op
        self.n_samples = n_samples
        self.bernoulli_p = bernoulli_p

        self.data_d = self.d
        self.seq_length = seq_length

        # a final sequence where at each step, the most recent 2 operands are repeated along with <EOS> tag
        # the time limit of unrolling is set to horizon number of steps
        self.set_horizon(horizon)

        np.random.seed(seed)

    def set_horizon(self, horizon):
        self.horizon = horizon
        self.final_seq_length = self.seq_length + self.horizon

    def __iter__(self):
        for i in range(self.n_samples):
            x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
            y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))

            u = torch.zeros((self.final_seq_length, self.batch_size, self.d))
            u[: self.seq_length] = self.initialization_op(
                self.seq_length, self.batch_size, self.d, p=self.bernoulli_p
            )

            # store the complete trajectory
            for time_idx in range(self.horizon):
                u[time_idx + self.seq_length] = self.composition_op(
                    u[time_idx : time_idx + self.seq_length], time_idx
                )

            u_temp = u[:, 0, :]

            # split trajectory to input and output phases
            x_sample[: self.seq_length] = u[: self.seq_length]
            y_sample[self.seq_length :] = u[self.seq_length :]

            yield x_sample, y_sample
