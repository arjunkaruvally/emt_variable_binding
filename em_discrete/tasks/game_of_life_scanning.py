import os
import sys

import torch
from torch.nn import functional as F
from enum import Enum
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from em_discrete.dataset.simpleIOPhaseDynamicalSystem import SimpleIOPhaseDynamicalSystem

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def composition_func(x, time_idx):
    # x1 = x.detach().clone()
    _, batch_size, input_dim = x.shape

    x1 = torch.zeros(input_dim + 2, batch_size, input_dim + 2)
    x1[1:-1, :, 1:-1] = x
    x1[0, :, 1:-1] = x[-1, :, :]
    x1[-1, :, 1:-1] = x[0, :, :]
    x1[1:-1, :, 0] = x[:, :, -1]
    x1[1:-1, :, -1] = x[:, :, 0]

    x1 = torch.permute(x1, dims=(1, 0, 2))
    x1 = torch.unsqueeze(x1, dim=1)

    kernel = torch.tensor([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]]).float()
    kernel = torch.unsqueeze(kernel, dim=0)
    kernel = torch.unsqueeze(kernel, dim=0)

    neighbour_matrix = F.conv2d(x1, kernel)

    output = torch.zeros((batch_size, input_dim))

    ## Game of life rules - with periodic boundary conditions
    # Living cells with two or three neighbours live. (Survival)
    output_temp = output[0, :]
    output[:, :] = x[0, :, :].detach().clone()

    # Living cells with fewer than two neighbours die. (Under-population)
    output[torch.logical_and(x[0, :, :].squeeze() == 1, (neighbour_matrix[:, 0, 0, :].squeeze() < 2))] = 0

    # Living cells with more than three neighbours die. (Overcrowding)
    output[torch.logical_and(x[0, :, :].squeeze() == 1, (neighbour_matrix[:, 0, 0, :].squeeze() > 3))] = 0

    # Dead cells with exactly three neighbours become alive. (Reproduction)
    output[torch.logical_and(x[0, :, :].squeeze() == 0, (neighbour_matrix[:, 0, 0, :].squeeze() == 3))] = 1

    return output


class GameOfLifeScanning(pl.LightningModule):
    def __init__(self, model, seed=0, learning_rate=1e-5, batch_size=32, input_dim=6,
                 model_type='rnn', poly_power=1.0, curriculum=True, l2_penalty=0.0,
                 bernoulli_p=0.3,
                 curriculum_threshold=0.96, composition_operation="xor", **kwargs):
        super(GameOfLifeScanning, self).__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length = input_dim
        self.poly_power = poly_power
        self.composition_operation = composition_operation
        self.l2_penalty = l2_penalty
        self.curriculum_threshold = curriculum_threshold
        self.bernoulli_p = bernoulli_p

        full_train_dataset = SimpleIOPhaseDynamicalSystem(composition_op=composition_func,
                                                          d=self.input_dim,
                                                          seq_length=input_dim,
                                                          bernoulli_p=self.bernoulli_p,
                                                          batch_size=self.batch_size,
                                                          seed=seed)
        full_val_dataset = SimpleIOPhaseDynamicalSystem(composition_op=composition_func,
                                                        d=self.input_dim,
                                                        seq_length=input_dim,
                                                        batch_size=self.batch_size,
                                                        bernoulli_p=self.bernoulli_p,
                                                        seed=seed)
        full_test_dataset = SimpleIOPhaseDynamicalSystem(composition_op=composition_func,
                                                         d=self.input_dim,
                                                         seq_length=input_dim,
                                                         batch_size=self.batch_size,
                                                         bernoulli_p=self.bernoulli_p,
                                                         seed=seed)

        self.train_dataset = full_train_dataset
        self.val_dataset = full_val_dataset
        self.test_dataset = full_test_dataset

        # curiculum learning parameters
        self.curriculum = curriculum
        self.curriculum_horizons = [3, 6, 9, 12, 15, 18]  # for ATOMIC_SIZE=1
        self.current_curriculum = 0

        if self.curriculum:
            self.train_dataset.set_horizon(self.curriculum_horizons[self.current_curriculum])

        # uncomment this to make the computational graph in tensorboard
        # self.example_input_array = torch.rand((self.seq_length, self.batch_size, self.input_dim + 2))

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        print("==================Logging computational graph")
        self.model.device = self.device
        self.model.initialize_hidden(self.batch_size, device=self.device)

    def training_step(self, batch, batch_nb):
        x, y = batch

        x = x.float().squeeze()
        y = y.float().squeeze()

        # temp_x = x[:, 0, :].squeeze().cpu().numpy()
        # temp_y = y[:, 0, :].squeeze().cpu().numpy()

        # convert inputs from 0/1 to -1/1
        x[:self.seq_length, :, :] = x[:self.seq_length, :, :] * 2 - 1
        y[self.seq_length:, :, :] = y[self.seq_length:, :, :] * 2 - 1

        # temp_grid = temp_x[:self.input_dim, :]
        # for seq_id in range(self.input_dim, temp_y.shape[1]):
        #     temp_grid[seq_id % self.input_dim, :] = temp_y[seq_id, :]

        self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
        y_hat = self.forward(x)

        y_hat = torch.stack(y_hat, dim=0).squeeze()
        y_hat = y_hat.reshape((-1, self.input_dim))
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat[self.seq_length:, :, :]

        # temp_y_hat = y_hat[:, 0, :].squeeze().detach().cpu().numpy()
        y_hat = y_hat.reshape((-1, self.input_dim))

        y = y[self.seq_length:, :, :]
        # temp_y = y[:, 0, :].squeeze().detach().cpu().numpy()
        y = y.reshape((-1, self.input_dim))

        loss = (y - y_hat) ** 2  # use the mse loss for
        loss = torch.mean(loss)

        # compute accuracy on only the y section with output
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y.reshape((-1, self.batch_size, self.input_dim))
        y = y.reshape((-1, self.input_dim))

        # convert predictions
        y_hat_predictions = y_hat.detach().clone()
        y_hat_predictions[y_hat_predictions >= 0] = 1.0
        y_hat_predictions[y_hat_predictions < 0] = -1.0
        y_hat_predictions = y_hat_predictions.long()
        y = y.long()

        y = y.cpu().detach().numpy()
        y_hat_predictions = y_hat_predictions.cpu().detach().numpy()

        accuracy = (y_hat_predictions.astype(int) == y.astype(int)).astype(int)
        accuracy = np.mean(accuracy)

        # increase difficulty of difficulty if the accuracy becomes greater than the curriculum accuracy
        if accuracy > self.curriculum_threshold:
            if self.curriculum and self.current_curriculum < len(self.curriculum_horizons) - 1:
                self.current_curriculum += 1
                # truncate curriculum
                self.current_curriculum = min(self.current_curriculum, len(self.curriculum_horizons) - 1)
                self.train_dataset.set_horizon(self.curriculum_horizons[self.current_curriculum])
                print("Curriculum difficulty increased to: {}/{}".format(self.current_curriculum + 1,
                                                                         len(self.curriculum_horizons)))

        self.log("training loss", loss.cpu().item())
        self.log("accuracy", accuracy, prog_bar=True)

        logs = {"loss": loss, "accuracy": accuracy}
        return logs

    def training_end(self, outputs):
        # print(outputs)
        outputs["avg_train_accuracy"] = 0
        return outputs

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        accuracy = (y_hat.argmax(1) == y).float().mean()

        return {'test_accuracy': accuracy}

    def test_end(self, outputs):
        avg_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()

        logs = {"test_accuracy": avg_accuracy}

        return {'test_accuracy': avg_accuracy, "log": logs}

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
        return self.optimizer

    def train_dataloader(self):
        print("-------------------------Initializing train dataloader")
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)
