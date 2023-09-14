import os
import sys

import torch
from torch.nn import functional as F
from enum import Enum
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class InputStream(Enum):
    EOB = 0
    DATASTREAM_START = 0
    EVALUATION_FROM = 2
    ATOMIC_SIZE = 3


class BinaryFibonacciNumbersDataset(IterableDataset):
    """Binary Logic dataset."""

    # NOTE: set horizon as multiple of whatever length atomic operation
    def __init__(self, seed=0, d=7, batch_size=64, horizon=36):
        self.d = d
        self.batch_size = batch_size
        self.horizon = horizon
        self.STREAM = InputStream
        self.data_d = self.d - self.STREAM.DATASTREAM_START.value

        # Sequence will consist of 2 seed operands with <EOB> tag
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
        self.final_seq_length = self.STREAM.EVALUATION_FROM.value + self.horizon

    def __iter__(self):
        for i in range(5000):
            x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
            y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))

            a = torch.randint(0, 2 ** self.data_d,
                              size=(self.batch_size, int(self.horizon / self.STREAM.ATOMIC_SIZE.value)))
            b = torch.randint(0, 2 ** self.data_d,
                              size=(self.batch_size, int(self.horizon / self.STREAM.ATOMIC_SIZE.value)))
            c = torch.zeros(size=(self.batch_size, int(self.horizon / self.STREAM.ATOMIC_SIZE.value))).long()

            for i in range(int(self.horizon / self.STREAM.ATOMIC_SIZE.value)):
                c[:, i] = a[:, i] + b[:, i]

                if i < int(self.horizon / self.STREAM.ATOMIC_SIZE.value) - 1:
                    a[:, i + 1] = b[:, i].clone()
                    b[:, i + 1] = c[:, i].clone()

            # if ((c / 2 ** self.data_d) > 1).any():
            #     print("Overflow")
            #     sys.exit()

            # pad with zeros
            z = torch.zeros(c.shape).long()
            number_stream = torch.stack((a, b, c), dim=2)
            number_stream = torch.flatten(number_stream, 1)
            number_stream = number_stream.view((-1, 1))
            binarized_stream = self.convert_to_binary(number_stream)
            binarized_stream = binarized_stream.view((self.batch_size, -1, self.data_d))
            binarized_stream = binarized_stream.permute((1, 0, 2))

            # store c
            y_sample[:, :, :] = 0
            y_sample[self.STREAM.EVALUATION_FROM.value:, :,
            self.STREAM.DATASTREAM_START.value:] = binarized_stream

            # add break indicator
            # y_sample[self.STREAM.EVALUATION_FROM.value::self.STREAM.ATOMIC_SIZE.value, :,
            # self.STREAM.EOB.value] = 1

            # x_sample[:self.STREAM.EVALUATION_FROM.value, :, self.STREAM.DATASTREAM_START.value:] = -1
            x_sample[:self.STREAM.EVALUATION_FROM.value, :, :] = -1
            x_sample[0, :, self.STREAM.DATASTREAM_START.value:] = self.convert_to_binary(a[:, 0:1])*2-1
            # x_sample[1, :, self.STREAM.EOB.value] = 1
            x_sample[1, :, self.STREAM.DATASTREAM_START.value:] = self.convert_to_binary(b[:, 0:1])*2-1

            yield x_sample, y_sample


class BinaryFibonacciNumbersTask(pl.LightningModule):
    def __init__(self, model, seed=0, learning_rate=1e-5, batch_size=32, input_dim=6, seq_length=5,
                 model_type='rnn', poly_power=1.0, curriculum=True, l2_penalty=0.0,
                 curriculum_threshold=0.96, **kwargs):
        super(BinaryFibonacciNumbersTask, self).__init__()
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
        full_train_dataset = BinaryFibonacciNumbersDataset(seed=self.seed,
                                                           d=self.input_dim,
                                                           batch_size=batch_size)
        full_val_dataset = BinaryFibonacciNumbersDataset(seed=self.seed,
                                                         d=self.input_dim,
                                                         batch_size=batch_size)
        full_test_dataset = BinaryFibonacciNumbersDataset(seed=self.seed,
                                                          d=self.input_dim,
                                                          batch_size=batch_size)
        self.train_dataset = full_train_dataset
        self.val_dataset = full_val_dataset
        self.test_dataset = full_test_dataset

        # curiculum learning parameters
        self.curriculum = curriculum
        # self.curriculum_horizons = [3, 6, 9, 12, 15, 18]  # for ATOMIC_SIZE=1
        # self.curriculum_horizons = [6, 8, 10, 12, 14, 16, 18, 20]  # for ATOMIC_SIZE=2
        self.curriculum_horizons = [9, 12, 15, 18, 21, 24, 27, 30]  # for ATOMIC_SIZE=3
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
        # y_hat = y_hat*2 - 1  # change domain to +1/-1
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat[self.train_dataset.STREAM.EVALUATION_FROM.value:, :, :]
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y[self.train_dataset.STREAM.EVALUATION_FROM.value:, :, :]
        y = y.reshape((-1, self.input_dim))

        loss = (y - y_hat) ** 2  # use the mse loss for
        loss = torch.mean(loss)

        # compute accuracy on only the y section with output
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat[self.train_dataset.STREAM.EVALUATION_FROM.value:, :, :]
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y.reshape((-1, self.batch_size, self.input_dim))
        y = y[self.train_dataset.STREAM.EVALUATION_FROM.value:, :, :]
        y = y.reshape((-1, self.input_dim))

        # convert predictions
        y_hat_predictions = y_hat.detach().clone()
        y_hat_predictions[y_hat_predictions >= 0.5] = 1
        y_hat_predictions[y_hat_predictions < 0.5] = 0
        y_hat_predictions = y_hat_predictions.long()

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
