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


class BinaryLinearVBDataset(IterableDataset):
    """
    Binary and Linear Variable Binding Task.
    each task is identified by a tuple of 3 numbers assigned to it assigned to it.

    (task_id_1, task_id_2, task_id_3)

    task_id_1 is the id for the power set enumeration of { 1, 2, 3, ... nd }.
    note that task_id_1 has to have d ones in its binary representation.

    task_id_2 is converted to a d-ary number encoding the combination of the choice obtained from the power set
    note that task_id_2 has to also satisfy certain conditions on the d-ary number. meaning the ith digit should be
    less than d-i.

    task_id_3 is binarized and represents the presence of absence of the sign in each element of the combination

    Note: id for repeat copy is always (2^{d}-1, 0, 0)
    """

    def __init__(
        self,
        seed=0,
        d=5,
        seq_length=10,
        batch_size=64,
        horizon=36,
        task_id=(31, 0, 0),
        generator_stop=2000,
    ):
        self.d = d
        self.batch_size = batch_size
        self.horizon = horizon
        self.seq_length = seq_length
        self.data_d = self.d
        self.composition_size = d // seq_length
        self.generator_stop = generator_stop

        self.f_operator = self.get_linear_oprerator(task_id)
        # plt.imshow(self.f_operator, cmap='coolwarm')
        # plt.show()

        self.set_horizon(horizon)

        np.random.seed(seed)

    def set_horizon(self, horizon):
        self.horizon = horizon
        self.final_seq_length = self.seq_length + self.horizon

    def get_linear_oprerator(self, task_id):
        task_id1, task_id2, task_id3 = task_id

        ## Step 1: convert task_id1 to the choice of dimensions
        # convert the binary representation of task_id1 to the choice of dimensions
        choice1 = []
        ndimensions = 0
        while task_id1:
            if task_id1 & 1:
                choice1.append(ndimensions)
            task_id1 >>= 1
            ndimensions += 1

        assert len(choice1) == self.d, (
            "task_id1 must have d ones in its binary representation. "
            "Check hamming weight of task_id1. Expected: {} Found: {} "
            "for task_id: {}".format(self.d, len(choice1), task_id)
        )

        ## Step 2: convert task_id2 to the combination of the choice
        choice_d = 0
        while task_id2:
            digit = task_id2 % self.d
            if digit > 0:
                assert digit < self.d - choice_d, (
                    "task_id2 must have the ith digit greater than d-i. "
                    "Check the choice of dimensions in task_id2"
                )
                # swap choice1[choice_d] and choice1[choice_d + digit]
                choice1[choice_d], choice1[choice_d + digit] = (
                    choice1[choice_d + digit],
                    choice1[choice_d],
                )

            task_id2 //= self.d
            choice_d += 1

        ## Step 3: convert task_id3 to the sign of the choice
        signs = [1] * self.d
        choice_d = 0
        while task_id3:
            digit = task_id3 % 2
            if digit == 1:
                signs[choice_d] = -1
            task_id3 //= 2
            choice_d += 1

        ## Step 4: create the linear operator
        linear_operator = torch.zeros((self.seq_length * self.d, self.d))
        for d, choice_d in enumerate(choice1):
            linear_operator[choice_d, d] = signs[d]

        return linear_operator

    def convert_from_binary(self, x):
        x = x.clone()
        x[x < 0] = 0
        mask = 2 ** torch.arange(self.data_d - 1, -1, -1).to(x.device, x.dtype)
        return torch.sum(mask * x, -1)

    def __iter__(self):
        for i in range(self.generator_stop):
            x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
            y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))

            x_sample[: self.seq_length, :, :] = (
                torch.randint(0, 2, size=(self.seq_length, self.batch_size, self.d)) * 2
                - 1
            )

            y_sample[:, :, :] = x_sample

            for step in range(self.horizon):
                composed_seq_id = self.seq_length + step

                u_t = None
                u_t = y_sample[
                    composed_seq_id - self.seq_length : composed_seq_id, :, :
                ]
                u_t = torch.swapaxes(u_t, 0, 1)
                u_t = u_t.reshape((self.batch_size, -1))

                # y_temp1 = y_sample[composed_seq_id - self.seq_length:composed_seq_id, 0, :].cpu().detach().data.numpy().copy()
                #
                # u_temp1 = u_t[0, :].cpu().detach().data.numpy().copy()

                u_t = u_t @ self.f_operator
                y_sample[composed_seq_id, :, :] = u_t

                # u_temp = u_t[0, :].cpu().detach().data.numpy()
                #
                # y_temp1 = y_sample[composed_seq_id - self.seq_length:composed_seq_id, 0,
                #           :].cpu().detach().data.numpy().copy()
                #
                # y_temp = y_sample[:, 0, :].cpu().data.numpy()

            # plt.imshow(y_temp, cmap='coolwarm')
            # plt.show()

            # y_sample = y_sample * 2 - 1
            y_sample[: self.seq_length, :, :] = 0
            # x_sample[:self.seq_length, :, :] = x_sample[:self.seq_length, :, :] * 2 - 1  # convert to in +1/-1 space

            yield x_sample, y_sample


class BinaryLinearVBTask(pl.LightningModule):
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
        task_id=(31, 0, 0),
        curriculum_threshold=0.96,
        **kwargs,
    ):
        super(BinaryLinearVBTask, self).__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.poly_power = poly_power
        self.l2_penalty = l2_penalty
        self.curriculum_threshold = curriculum_threshold

        # splitting datasets here so training and validation do not overlap
        full_train_dataset = BinaryLinearVBDataset(
            seed=self.seed,
            d=self.input_dim,
            seq_length=self.seq_length,
            task_id=task_id,
            batch_size=batch_size,
        )
        full_val_dataset = BinaryLinearVBDataset(
            seed=self.seed,
            seq_length=self.seq_length,
            d=self.input_dim,
            task_id=task_id,
            batch_size=batch_size,
        )
        full_test_dataset = BinaryLinearVBDataset(
            seed=self.seed,
            d=self.input_dim,
            seq_length=self.seq_length,
            task_id=task_id,
            batch_size=batch_size,
        )
        self.train_dataset = full_train_dataset
        self.val_dataset = full_val_dataset
        self.test_dataset = full_test_dataset

        # curiculum learning parameters
        self.curriculum = curriculum
        self.curriculum_horizons = [10, 20, 40, 60, 100]
        # self.curriculum_horizons = [100]
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
        y_nums = (
            self.train_dataset.convert_from_binary(y[:, 0, :].squeeze())
            .detach()
            .cpu()
            .numpy()
            .reshape((-1, 1))
        )
        # plt.subplot(121)
        # plt.imshow(x[:, 0, :].squeeze().numpy(), cmap='coolwarm')
        # plt.subplot(122)
        # plt.imshow(y[:, 0, :].squeeze().numpy(), cmap='coolwarm')
        #
        # plt.show()

        self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
        y_hat = self.forward(x)

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

        accuracy = (y_hat_predictions.astype(np.int_) == y.astype(np.int_)).astype(
            np.int_
        )
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
