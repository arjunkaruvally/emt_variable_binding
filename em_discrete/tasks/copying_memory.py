import os
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CopyingMemoryDataset(IterableDataset):
    """TFA dataset."""

    def __init__(self, seed=0, d=10, seq_length=10, batch_size=64, T=50):
        self.d = d
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.T = T
        self.category_classes = torch.eye(d)

        self.final_seq_length = self.T + self.seq_length * 2

        np.random.seed(seed)

    def __iter__(self):
        x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d + 2))  # +2 is for the extra signals
        y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d + 2))  # +2 is for the extra signals

        # add filler signal
        x_sample[self.seq_length: self.seq_length + self.T - 1:, :, self.d] = 1
        x_sample[self.seq_length + self.T:, :, self.d] = 1
        y_sample[:self.T + self.seq_length, :, self.d] = 1

        # add end signal
        x_sample[self.seq_length + self.T - 1, :, self.d + 1] = 1

        # add actual signal
        for batch_id in range(self.batch_size):
            category_list = torch.LongTensor(np.random.choice(range(self.d), self.seq_length))

            x_sample[:self.seq_length, batch_id, :] = F.one_hot(category_list, num_classes=self.d + 2)
            y_sample[self.T + self.seq_length:, batch_id, :] = F.one_hot(category_list, num_classes=self.d + 2)

        yield x_sample, y_sample


class CopyingMemoryTask(pl.LightningModule):
    def __init__(self, model, seed=0, learning_rate=1e-5, batch_size=32, input_dim=8, seq_length=10, T=100,
                 model_type='rnn', poly_power=1.0, **kwargs):
        super(CopyingMemoryTask, self).__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.poly_power = poly_power
        self.T = T

        # splitting datasets here so training and validation do not overlap
        full_train_dataset = CopyingMemoryDataset(seed=self.seed,
                                                  d=self.input_dim,
                                                  batch_size=batch_size,
                                                  seq_length=self.seq_length,
                                                  T=T)
        full_val_dataset = CopyingMemoryDataset(seed=self.seed,
                                                d=self.input_dim,
                                                batch_size=batch_size,
                                                seq_length=self.seq_length,
                                                T=T)
        full_test_dataset = CopyingMemoryDataset(seed=self.seed,
                                                 d=self.input_dim,
                                                 batch_size=batch_size,
                                                 seq_length=self.seq_length,
                                                 T=T)
        self.train_dataset = full_train_dataset
        self.val_dataset = full_val_dataset
        self.test_dataset = full_test_dataset

        # uncomment this to make the computational graph in tensorboard
        # self.example_input_array = torch.rand((self.seq_length, self.batch_size, self.input_dim + 2))

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        print("==================Logging computational graph")
        self.model.device = self.device
        self.model.initialize_hidden(self.batch_size, device=self.device)

    def validation_step(self, batch, batch_nb):
        with torch.enable_grad():
            x, y = batch
            x = x.float().squeeze()
            y = y.float().squeeze()
            x.requires_grad = True

            y_orig = y.clone()

            # print(key)
            # for key_idx, key in enumerate([ 'epHop_1.0' ]):
            self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
            x.requires_grad = True
            y_hat = self.model.forward(x)

            y_hat = y_hat[self.T + self.seq_length + 1].squeeze()
            y = y_orig[self.T + self.seq_length + 1, :, :]

            loss = F.cross_entropy(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            xgrad = x.grad

            norms = []
            lookbacks = []

            for i in range(0, self.T + 20):
                # print(i)
                lookbacks.append(i)
                norms.append(torch.linalg.vector_norm(xgrad[i, :, :].squeeze(), dim=1).mean().item())
            self.optimizer.zero_grad()

        self.logger.experiment.add_histogram_raw("x_grad norms",
                                                 np.min(norms),
                                                 np.max(norms),
                                                 len(norms),
                                                 np.sum(norms),
                                                 np.sum(np.array(norms)**2),
                                                 np.arange(0, len(norms), 1),
                                                 np.array(norms), global_step=self.global_step)

    def training_step(self, batch, batch_nb):
        x, y = batch

        x = x.float().squeeze()
        y = y.float().squeeze()

        self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
        y_hat = self.forward(x)

        y_hat = torch.stack(y_hat, dim=0).squeeze()

        y = y.squeeze()
        y = y.reshape((-1, self.input_dim + 2))
        y_hat = y_hat.reshape((-1, self.input_dim + 2))
        # y_hat = torch.sigmoid(y_hat)

        # careful - y_hat is the UNNORMALIZED logits
        loss = F.cross_entropy(y_hat, y.argmax(1))
        # loss = y * torch.log(torch.clamp(y_hat, 1e-10))

        accuracy = (y_hat.argmax(1)[-self.seq_length * self.batch_size:] == y.argmax(1)[
                                                                            -self.seq_length * self.batch_size:]).float().mean()
        # y_hat = y_hat.reshape((self.seq_length, self.batch_size, self.input_dim))
        # y = y.reshape((self.seq_length, self.batch_size, self.input_dim))

        logs = {"loss": loss}
        self.log("training loss", logs)

        logs = {"accuracy": accuracy}
        self.log("accuracy", logs, prog_bar=True)

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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer

    def train_dataloader(self):
        print("-------------------------Initializing train dataloader")
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)
