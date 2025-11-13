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


class InputStream(Enum):
    LAND = 0
    LOR = 1
    DATALINE_START = 2


class BinaryArithmeticDataset(IterableDataset):
    """
    Binary Arithmetic dataset.
    seq_length sets of numbers are subject to arithmetic and logical operations
    """

    def __init__(self, seed=0, d=9, batch_size=64, repeats=4):
        self.d = d
        self.batch_size = batch_size
        self.repeats = repeats

        self.STREAM = InputStream
        self.operand_d = d - int(self.STREAM.DATALINE_START.value)

        # input sequence will consist of 2 operands followed by an operator (total 3 steps)
        # output sequence is a string of zeros followed by repeat copy of operand+operator+result
        self.final_seq_length = 3 + 4 * (1 + repeats)

        np.random.seed(seed)

    def convert_to_binary(self, x):
        mask = 2 ** torch.arange(self.operand_d).to(x.device, x.dtype)
        mask = torch.repeat_interleave(mask.reshape((1, -1)), self.batch_size, dim=0)

        return x.bitwise_and(mask).ne(0).long()

    def __iter__(self):
        for i in range(10000):
            x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
            y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))

            a = torch.randint(0, 2 ** (self.operand_d), size=(self.batch_size, 1))
            b = torch.randint(0, 2 ** (self.operand_d), size=(self.batch_size, 1))
            c_and = torch.bitwise_and(a, b)
            c_or = torch.bitwise_or(a, b)

            and_mask = torch.randint(0, 2, size=(self.batch_size, 1))

            # add input data to the input stream
            x_sample[0, :, self.STREAM.DATALINE_START.value :] = self.convert_to_binary(
                a
            )
            x_sample[1, :, self.STREAM.DATALINE_START.value :] = self.convert_to_binary(
                b
            )

            # add operator signal
            x_sample[2, :, self.STREAM.LAND.value] = and_mask.flatten()
            x_sample[2, :, self.STREAM.LOR.value] = 1 - and_mask.flatten()

            # add logical result to output
            input_offset = 3
            y_sample[input_offset : input_offset + 3, :, :] = x_sample[:3, :, :]
            y_sample[input_offset + 3, :, self.STREAM.DATALINE_START.value :] = (
                self.convert_to_binary(and_mask * c_and + (1 - and_mask) * c_or)
            )

            # add repeats
            if self.repeats > 0:
                y_sample[input_offset + 4 :] = y_sample[
                    input_offset : input_offset + 4
                ].tile((self.repeats, 1, 1))

            yield x_sample, y_sample


class BinaryArithmeticTask(pl.LightningModule):
    def __init__(
        self,
        model,
        seed=0,
        learning_rate=1e-5,
        batch_size=32,
        input_dim=6,
        seq_length=5,
        model_type="rnn",
        poly_power=1.0,
        **kwargs,
    ):
        super(BinaryArithmeticTask, self).__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.poly_power = poly_power

        # splitting datasets here so training and validation do not overlap
        full_train_dataset = BinaryArithmeticDataset(
            seed=self.seed, d=self.input_dim, batch_size=batch_size
        )
        full_val_dataset = BinaryArithmeticDataset(
            seed=self.seed, d=self.input_dim, batch_size=batch_size
        )
        full_test_dataset = BinaryArithmeticDataset(
            seed=self.seed, d=self.input_dim, batch_size=batch_size
        )
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

    def training_step(self, batch, batch_nb):
        x, y = batch

        x = x.float().squeeze()
        y = y.float().squeeze()
        temp_x = x[:, 0, :].squeeze().cpu().numpy()
        temp_y = y[:, 0, :].squeeze().cpu().numpy()
        # plt.subplot(121)
        # plt.imshow(x[:, 0, :].squeeze().numpy(), cmap='coolwarm')
        # plt.subplot(122)
        # plt.imshow(y[:, 0, :].squeeze().numpy(), cmap='coolwarm')
        #
        # plt.show()

        self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
        y_hat = self.forward(x)

        y_hat = torch.stack(y_hat, dim=0).squeeze()
        y_hat = y_hat.reshape((-1, self.input_dim))
        y_hat = torch.sigmoid(y_hat)  # constraint the output dimension to [0, 1]
        y = y.reshape((-1, self.input_dim))

        loss = (y - y_hat) ** 2  # use the mse loss for
        loss = torch.sum(loss)

        # compute accuracy on only the y section with output
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat[3:, :, :]
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y.reshape((-1, self.batch_size, self.input_dim))
        y = y[3:, :, :]
        y = y.reshape((-1, self.input_dim))

        acc_temp = (y_hat > 0.5).long().flatten() == y.float().flatten().float()
        accuracy = (
            ((y_hat > 0.5).long().flatten() == y.float().flatten()).float().mean()
        )

        logs = {"loss": loss}
        self.log("training loss", loss.cpu().item())

        logs = {"accuracy": accuracy}
        self.log("accuracy", accuracy.cpu().item(), prog_bar=True)

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

        return {"test_accuracy": accuracy}

    def test_end(self, outputs):
        avg_accuracy = torch.stack([x["test_accuracy"] for x in outputs]).mean()

        logs = {"test_accuracy": avg_accuracy}

        return {"test_accuracy": avg_accuracy, "log": logs}

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
