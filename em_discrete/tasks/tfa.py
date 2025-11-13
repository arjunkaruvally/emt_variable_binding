import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset


class TFADataset(IterableDataset):
    """TFA dataset."""

    def __init__(self, seed=0, P=6, symbol_prob=0.9, batch_size=32, transform=None):
        self.P = P
        self.seed = seed
        self.symbol_prob = symbol_prob
        self.batch_size = batch_size

        np.random.seed(seed)

        self.cur_state = 1
        self.cur_tod = 0
        self.t = 0

        self.transition_dict = {
            0: {  # day transition logic
                1: {"a": 1, "b": 2, "c": 3},
                2: {"a": 1, "b": 2, "c": 3},
                3: {"a": 1, "b": 2, "c": 3},
            },
            1: {  # night transition logic
                1: {"a": 1, "b": 3, "c": 2},
                2: {"a": 1, "b": 3, "c": 2},
                3: {"a": 1, "b": 3, "c": 2},
            },
        }

        self.symbol_map = {
            None: np.array([0.0, 0.0, 0.0]),
            "a": np.array([0.0, 0.0, 1.0]),
            "b": np.array([0.0, 1.0, 0.0]),
            "c": np.array([1.0, 0.0, 0.0]),
        }
        self.state_map = {
            1: np.array([0]),
            2: np.array([1]),
            3: np.array([2]),
        }

        self.transform = transform

    def transition(self, in_symbol=None):
        if in_symbol is not None:
            self.cur_state = self.transition_dict[self.cur_tod][self.cur_state][
                in_symbol
            ]

        self.t = (self.t + 1) % self.P
        self.cur_tod = int(self.t >= int(self.P / 2))

        return self.cur_state

    def __iter__(self):
        in_symbols = np.random.choice(
            [None, "a", "b", "c"],
            p=[
                1 - self.symbol_prob,
                self.symbol_prob / 3,
                self.symbol_prob / 3,
                self.symbol_prob / 3,
            ],
            size=self.batch_size,
        )

        sample = np.array([self.symbol_map[in_symbol] for in_symbol in in_symbols])
        y_sample = np.array(
            [self.state_map[self.transition(in_symbol)] for in_symbol in in_symbols]
        )

        if self.transform:
            sample = self.transform(sample)
            y_sample = self.transform(y_sample)

        yield sample, y_sample


class TFA(pl.LightningModule):
    def __init__(
        self, model, seed=0, learning_rate=1e-5, batch_size=32, reset_hidden=False
    ):
        super(TFA, self).__init__()
        self.seed = seed
        self.reset_hidden = reset_hidden
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # splitting datasets here so training and validation do not overlap
        full_train_dataset = TFADataset(seed=self.seed, batch_size=batch_size)
        val_nb = TFADataset(seed=1)
        tng_nb = TFADataset(seed=2)
        self.train_dataset = full_train_dataset
        self.val_dataset = full_train_dataset
        self.test_dataset = full_train_dataset

        self.model.initialize_hidden(batch_size=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch

        x = x.reshape(-1, self.batch_size, 3)
        x = x.float()

        y_hat = self.forward(x).squeeze()
        y_hat = F.softmax(y_hat, dim=1)
        y = y.squeeze()

        if self.reset_hidden:
            self.model.initialize_hidden(batch_size=1)

        loss = F.cross_entropy(y_hat, y)

        accuracy = (y_hat.argmax(1) == y).float().mean()

        logs = {"loss": loss, "train_accuracy": accuracy}

        if not self.reset_hidden:
            self.model.h = self.model.h.detach()

        self.logger.experiment.add_scalar("loss", loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "train_accuracy", accuracy, self.current_epoch
        )

        return {"loss": loss, "train_accuracy": accuracy, "log": logs}

    def training_end(self, outputs):
        # print(outputs)
        outputs["avg_train_accuracy"] = 0
        return outputs

    def validation_step(self, batch, batch_nb):
        pass
        # x, y = batch
        #
        # x = x.reshape(-1, self.batch_size, 3)
        # x = x.float()
        #
        # y_hat = self.forward(x).squeeze()
        # y = y.squeeze()
        #
        # loss = F.cross_entropy(y_hat, y)
        # accuracy = (y_hat.argmax(1) == y).float().mean()
        #
        # return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_end(self, outputs):
        pass
        # avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #
        # logs = {
        #     "validation_accuracy": avg_accuracy,
        #     "validation_loss": avg_loss
        # }
        #
        # return {'val_accuracy': avg_accuracy, "log": logs}

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)
