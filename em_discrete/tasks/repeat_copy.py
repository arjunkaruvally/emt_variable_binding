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


class RepeatCopyDataset(IterableDataset):
    """Repeat Copy dataset."""

    def __init__(self, seed=0, d=7, seq_length=5, batch_size=64, repeats=4, T=50):
        self.d = d
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.T = T
        self.category_classes = torch.eye(d)
        self.repeats = repeats

        # Sequence will consist of 'seq_length' number of vectors of 'd' dimensions
        # a final EOS tag and then no input
        # output - repeat copies the input vectors atleast 'repeats' times
        self.final_seq_length = self.seq_length * (repeats + 1)

        np.random.seed(seed)

    def convert_to_binary(self, x):
        mask = 2 ** torch.arange(self.seq_length).to(x.device, x.dtype)
        mask = torch.repeat_interleave(mask.reshape((1, -1)), self.batch_size, dim=0)

        return x.bitwise_and(mask).ne(0).long()

    def __iter__(self):
        for i in range(2000):
            x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
            y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))

            # construct input signal
            # add the input indicator signal
            x_sample[: self.seq_length, :, :] = (
                torch.randint(0, 2, size=(self.seq_length, self.batch_size, self.d)) * 2
                - 1
            )

            # construct output signal
            # add input data to the input stream 5 times
            y_sample[self.seq_length :, :, :] = x_sample[: self.seq_length, :, :].tile(
                (self.repeats, 1, 1)
            )

            yield x_sample, y_sample


class RepeatCopyTask(pl.LightningModule):
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
        l2_penalty=0.0,
        **kwargs,
    ):
        super(RepeatCopyTask, self).__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.poly_power = poly_power
        self.l2_penalty = l2_penalty

        # splitting datasets here so training and validation do not overlap
        full_train_dataset = RepeatCopyDataset(
            seed=self.seed,
            d=self.input_dim,
            batch_size=batch_size,
            seq_length=self.seq_length,
        )
        full_val_dataset = RepeatCopyDataset(
            seed=self.seed,
            d=self.input_dim,
            batch_size=batch_size,
            seq_length=self.seq_length,
        )
        full_test_dataset = RepeatCopyDataset(
            seed=self.seed,
            d=self.input_dim,
            batch_size=batch_size,
            seq_length=self.seq_length,
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

        # print(y_hat.shape)
        # y_hat = torch.stack(y_hat, dim=0).squeeze()
        y_hat = y_hat.reshape((-1, self.input_dim))
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat[self.seq_length :, :, :]
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y[self.seq_length :, :, :]
        y = y.reshape((-1, self.input_dim))

        loss = (y - y_hat) ** 2  # use the mse loss for
        loss = torch.mean(loss)

        # compute accuracy on only the y section with output
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        # y_hat = y_hat[self.train_dataset.STREAM.EVALUATION_FROM.value:, :, :]
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y.reshape((-1, self.batch_size, self.input_dim))
        # y = y[self.train_dataset.STREAM.EVALUATION_FROM.value:, :, :]
        y = y.reshape((-1, self.input_dim))

        # convert predictions
        y_hat_predictions = y_hat.detach().clone()
        y_hat_predictions[y_hat_predictions >= 0] = 1
        y_hat_predictions[y_hat_predictions < 0] = -1
        y_hat_predictions = y_hat_predictions.long()

        y = y.cpu().detach().numpy()
        y_hat_predictions = y_hat_predictions.cpu().detach().numpy()

        accuracy = (y_hat_predictions.astype(np.int64) == y.astype(np.int64)).astype(
            np.int64
        )
        accuracy = np.mean(accuracy)

        self.log("training loss", loss.cpu().item(), prog_bar=True)
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

        return {"test_accuracy": accuracy}

    def test_end(self, outputs):
        avg_accuracy = torch.stack([x["test_accuracy"] for x in outputs]).mean()

        logs = {"test_accuracy": avg_accuracy}

        return {"test_accuracy": avg_accuracy, "log": logs}

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty
        )
        return self.optimizer

    def train_dataloader(self):
        print("-------------------------Initializing train dataloader")
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)
