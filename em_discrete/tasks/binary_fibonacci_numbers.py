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


class BinaryFibonacciNumbersDataset(IterableDataset):
    """Binary Logic dataset."""

    # NOTE: set horizon as multiple of whatever length atomic operation
    def __init__(self, seed=0, d=7, batch_size=64, seq_length=2, horizon=36, composition_operation="add"):
        self.d = d
        self.batch_size = batch_size
        self.horizon = horizon
        self.composition_operation = composition_operation
        self.data_d = self.d
        self.seq_length = seq_length
        self.ATOMIC_SIZE = 2

        # a final sequence where at each step, the most recent 2 operands are repeated along with <EOS> tag
        # the time limit of unrolling is set to horizon number of steps
        self.set_horizon(horizon)

        np.random.seed(seed)

    def convert_to_binary(self, x):
        mask = 2 ** torch.arange(self.data_d).to(x.device, x.dtype)
        mask = torch.repeat_interleave(mask.reshape((1, -1)), x.shape[0], dim=0)

        return x.bitwise_and(mask).ne(0).long()

    def convert_from_binary(self, x):
        mask = 2 ** torch.arange(self.data_d - 1, -1, -1).to(x.device, x.dtype)
        return torch.sum(mask * x, -1)

    def set_horizon(self, horizon):
        self.horizon = horizon
        self.final_seq_length = 2 * self.seq_length + self.horizon

    def __iter__(self):
        for i in range(5000):
            x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
            y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))

            a = torch.randint(0, 2 ** self.data_d,
                              size=(self.seq_length, self.batch_size, 1))
            c = torch.zeros(size=(self.batch_size, int(self.horizon / self.ATOMIC_SIZE))).long()

            # store a, b
            for var_id in range(self.seq_length):
                x_sample[var_id * 2, :, :] = self.convert_to_binary(a[var_id, :, :]) * 2 - 1

            for i in range(int(self.horizon / self.ATOMIC_SIZE)):
                res = a[0, :, :]
                if self.composition_operation == "add":
                    for var_id in range(1, self.seq_length):
                        res = res + a[var_id, :, :]
                elif self.composition_operation == "xor":
                    # c[:, i] = torch.bitwise_xor(a[:, 0], b[:, 0])
                    for var_id in range(1, self.seq_length):
                        res = torch.bitwise_xor(res, a[var_id, :, :])

                c[:, i] = res

                if i < int(self.horizon / self.ATOMIC_SIZE) - 1:
                    a[:, 0] = b[:, 0]
                    b[:, 0] = c[:, i]

            number_stream = c
            number_stream = torch.flatten(number_stream, 1)
            number_stream = number_stream.view((-1, 1))
            binarized_stream = self.convert_to_binary(number_stream)
            binarized_stream = binarized_stream.view((self.batch_size, -1, self.data_d))
            binarized_stream = binarized_stream.permute((1, 0, 2))

            # store c
            y_sample[:, :, :] = 0
            y_sample[self.seq_length * 2::2, :, :] = binarized_stream * 2 - 1

            yield x_sample, y_sample


class BinaryFibonacciNumbersTask(pl.LightningModule):
    def __init__(self, model, seed=0, learning_rate=1e-5, batch_size=32, input_dim=6, seq_length=5,
                 model_type='rnn', poly_power=1.0, curriculum=True, l2_penalty=0.0,
                 curriculum_threshold=0.96, composition_operation="xor", **kwargs):
        super(BinaryFibonacciNumbersTask, self).__init__()
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

        def composition_func(x, time_idx):
            if composition_operation == "xor":
                res = x[0, :, :]
                for i in range(1, x.shape[0]):
                    res = torch.logical_xor(res, x[i, :, :])
                return res
                # x.index_reduce_(0, torch.FloatTensor(range(x.shape[0])), x, torch.bitwise_xor)

        full_train_dataset = SimpleIOPhaseDynamicalSystem(composition_op=composition_func,
                                                          d=self.input_dim,
                                                          seq_length=seq_length,
                                                          batch_size=self.batch_size,
                                                          seed=seed)
        full_val_dataset = SimpleIOPhaseDynamicalSystem(composition_op=composition_func,
                                                        d=self.input_dim,
                                                        seq_length=seq_length,
                                                        batch_size=self.batch_size,
                                                        seed=seed)
        full_test_dataset = SimpleIOPhaseDynamicalSystem(composition_op=composition_func,
                                                         d=self.input_dim,
                                                         seq_length=seq_length,
                                                         batch_size=self.batch_size,
                                                         seed=seed)

        self.train_dataset = full_train_dataset
        self.val_dataset = full_val_dataset
        self.test_dataset = full_test_dataset

        # curiculum learning parameters
        self.curriculum = curriculum
        # self.curriculum_horizons = [3, 6, 9, 12, 15, 18]  # for ATOMIC_SIZE=1
        self.curriculum_horizons = [6, 8, 10, 12, 14, 16, 18, 20]  # for ATOMIC_SIZE=2
        # self.curriculum_horizons = [9, 12, 15, 18, 21, 24, 27, 30]  # for ATOMIC_SIZE=3
        # self.curriculum_horizons = [12, 16, 20, 24, 28, 32, 36, 40]  # for ATOMIC_SIZE=4
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

        temp_x = x[:, 0, :].squeeze().cpu().numpy()
        temp_y = y[:, 0, :].squeeze().cpu().numpy()

        # convert inputs from 0/1 to -1/1
        x[:self.seq_length, :, :] = x[:self.seq_length, :, :] * 2 - 1
        y[self.seq_length:, :, :] = y[self.seq_length:, :, :] * 2 - 1

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
        # y_hat = y_hat*2 - 1  # change domain to +1/-1
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat[self.seq_length:, :, :]

        temp_y_hat = y_hat[:, 0, :].squeeze().detach().cpu().numpy()
        y_hat = y_hat.reshape((-1, self.input_dim))

        y = y[self.seq_length:, :, :]
        temp_y = y[:, 0, :].squeeze().detach().cpu().numpy()
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
        y_hat_predictions[y_hat_predictions >= 0] = 1.0
        y_hat_predictions[y_hat_predictions < 0] = -1.0
        y_hat_predictions = y_hat_predictions.long()
        y = y.long()

        y = y.cpu().detach().numpy()
        y_hat_predictions = y_hat_predictions.cpu().detach().numpy()

        accuracy = (y_hat_predictions.astype(np.int) == y.astype(np.int)).astype(np.int)
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
