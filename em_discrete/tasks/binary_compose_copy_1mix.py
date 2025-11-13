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


class BinaryComposeCopyDataset(IterableDataset):
    """Binary Compose Copy dataset."""

    def __init__(self, seed=0, d=10, seq_length=5, batch_size=64, horizon=36):
        self.d = d
        self.batch_size = batch_size
        self.horizon = horizon
        self.seq_length = seq_length
        self.data_d = self.d
        self.composition_size = d // seq_length

        # Sequence will consist of seq_length number of seed operands
        # a final sequence where at each step, the last d/seq_length bits of each operand is composed to form a new one
        # this process is repeated till infinity
        # the time limit of unrolling is set to horizon number of steps
        self.set_horizon(horizon)

        np.random.seed(seed)

    def set_horizon(self, horizon):
        self.horizon = horizon
        self.final_seq_length = self.seq_length + self.horizon

    def __iter__(self):
        for i in range(2000):
            x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
            y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))

            x_sample[: self.seq_length, :, :] = (
                torch.randint(0, 2, size=(self.seq_length, self.batch_size, self.d)) * 2
                - 1
            )

            y_sample[:, :, :] = x_sample

            for step in range(self.horizon):
                composed_seq_id = self.seq_length + step
                compose_seq_id_1 = step
                compose_seq_id_2 = step + 1

                y_sample[composed_seq_id, :, : self.data_d // 2] = y_sample[
                    compose_seq_id_1, :, : self.data_d // 2
                ]
                y_sample[composed_seq_id, :, self.data_d // 2 :] = -y_sample[
                    compose_seq_id_2, :, -self.data_d // 2 :
                ]

            # y_sample = y_sample * 2 - 1
            y_sample[: self.seq_length, :, :] = 0
            # x_sample[:self.seq_length, :, :] = x_sample[:self.seq_length, :, :] * 2 - 1  # convert to in +1/-1 space

            yield x_sample, y_sample


class BinaryComposeCopyTask(pl.LightningModule):
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
        curriculum=True,
        l2_penalty=0.0,
        curriculum_threshold=0.96,
        composition_operation="add",
        **kwargs,
    ):
        super(BinaryComposeCopyTask, self).__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.poly_power = poly_power
        self.composition_operation = composition_operation
        self.l2_penalty = l2_penalty
        self.curriculum_threshold = curriculum_threshold

        # splitting datasets here so training and validation do not overlap
        full_train_dataset = BinaryComposeCopyDataset(
            seed=self.seed,
            d=self.input_dim,
            seq_length=self.seq_length,
            batch_size=batch_size,
        )
        full_val_dataset = BinaryComposeCopyDataset(
            seed=self.seed,
            seq_length=self.seq_length,
            d=self.input_dim,
            batch_size=batch_size,
        )
        full_test_dataset = BinaryComposeCopyDataset(
            seed=self.seed,
            d=self.input_dim,
            seq_length=self.seq_length,
            batch_size=batch_size,
        )
        self.train_dataset = full_train_dataset
        self.val_dataset = full_val_dataset
        self.test_dataset = full_test_dataset

        # curiculum learning parameters
        self.curriculum = curriculum
        self.curriculum_horizons = [10, 20, 40, 60, 100]
        self.current_curriculum = 0

        if self.curriculum:
            self.train_dataset.set_horizon(
                self.curriculum_horizons[self.current_curriculum]
            )

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
        # y_hat = torch.sigmoid(y_hat)  # constraint the output dimension to [0, 1]
        # y_hat = y_hat*2 - 1  # change domain to +1/-1
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

        accuracy = (y_hat_predictions.astype(np.int) == y.astype(np.int)).astype(np.int)
        accuracy = np.mean(accuracy)

        # increase difficulty of difficulty if the accuracy becomes greater than the curriculum accuracy
        if accuracy > self.curriculum_threshold:
            if (
                self.curriculum
                and self.current_curriculum < len(self.curriculum_horizons) - 1
            ):
                self.current_curriculum += 1
                # truncate curriculum
                self.current_curriculum = min(
                    self.current_curriculum, len(self.curriculum_horizons) - 1
                )
                self.train_dataset.set_horizon(
                    self.curriculum_horizons[self.current_curriculum]
                )
                print(
                    "Curriculum difficulty increased to: {}/{}".format(
                        self.current_curriculum + 1, len(self.curriculum_horizons)
                    )
                )

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
