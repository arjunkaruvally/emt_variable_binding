import os
import sys

import torch
from torch.nn import functional as F
from enum import Enum
from torch.utils.data import DataLoader, IterableDataset
from GameOfLife import NumpyWorld

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


w = NumpyWorld()
w.addPattern('glider')

while True:
    w.step()

# class InputStream(Enum):
#     DATALINE = 0
#     IN1_INDICATOR = 1
#     IN2_INDICATOR = 2
#     LAND = 3
#     LOR = 4
#     EOI = 5
#     EOS = 6
#
#
# class BinaryLogicDataset(IterableDataset):
#     """Binary Logic dataset."""
#
#     def __init__(self, seed=0, d=7, seq_length=5, batch_size=64, T=50):
#         self.d = d
#         self.seq_length = seq_length
#         self.batch_size = batch_size
#         self.T = T
#         self.category_classes = torch.eye(d)
#
#         self.STREAM = InputStream
#
#         # Sequence will consist of 2 operands and a result (total 3 seq lengths)
#         # a final operator (AND/OR) and EOI for input and EOS for output (3 steps)
#         self.final_seq_length = self.seq_length * 5 + 4
#
#         np.random.seed(seed)
#
#     def convert_to_binary(self, x):
#         mask = 2 ** torch.arange(self.seq_length).to(x.device, x.dtype)
#         mask = torch.repeat_interleave(mask.reshape((1, -1)), self.batch_size, dim=0)
#
#         return x.bitwise_and(mask).ne(0).long()
#
#     def __iter__(self):
#         for i in range(5000):
#             x_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
#             y_sample = torch.zeros((self.final_seq_length, self.batch_size, self.d))
#
#             # add the input indicator signal
#             x_sample[:self.seq_length, :, self.STREAM.IN1_INDICATOR.value] = 1
#             x_sample[self.seq_length:2 * self.seq_length, :, self.STREAM.IN2_INDICATOR.value] = 1
#
#             a = torch.randint(0, 2**self.seq_length, size=(self.batch_size, 1))
#             b = torch.randint(0, 2 ** self.seq_length, size=(self.batch_size, 1))
#             c_and = torch.bitwise_and(a, b)
#             c_or = torch.bitwise_or(a, b)
#
#             and_mask = torch.randint(0, 2, size=(self.batch_size, 1))
#
#             # construct input signal
#             # add input data to the input stream
#             x_sample[:self.seq_length, :, self.STREAM.DATALINE.value] = self.convert_to_binary(a).T
#             x_sample[self.seq_length:2*self.seq_length, :, self.STREAM.DATALINE.value] = self.convert_to_binary(b).T
#
#             # add operator signal
#             x_sample[2*self.seq_length, :, self.STREAM.LAND.value] = and_mask.flatten()
#             x_sample[2 * self.seq_length, :, self.STREAM.LOR.value] = 1 - and_mask.flatten()
#             x_sample[2*self.seq_length+1, :, self.STREAM.EOI.value] = 1
#
#             # construct output signal
#             # repeat input first
#             input_offset = 2*self.seq_length+2
#             y_sample[input_offset:input_offset+2*self.seq_length+1] = x_sample[:2*self.seq_length+1]
#
#             # add logical result to output
#             input_offset = 4*self.seq_length + 3
#             y_sample[input_offset:-1, :, self.STREAM.DATALINE.value] = self.convert_to_binary(and_mask*c_and +
#                                                                                             (1-and_mask)*c_or).T
#             y_sample[-1, :, self.STREAM.EOS.value] = 1
#
#             yield x_sample, y_sample
#
#
# class BinaryLogicTask(pl.LightningModule):
#     def __init__(self, model, seed=0, learning_rate=1e-5, batch_size=32, input_dim=6, seq_length=5,
#                  model_type='rnn', poly_power=1.0, **kwargs):
#         super(BinaryLogicTask, self).__init__()
#         self.save_hyperparameters()
#         self.seed = seed
#         self.model = model
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.input_dim = input_dim
#         self.seq_length = seq_length
#         self.poly_power = poly_power
#
#         # splitting datasets here so training and validation do not overlap
#         full_train_dataset = BinaryLogicDataset(seed=self.seed,
#                                                 d=self.input_dim,
#                                                 batch_size=batch_size,
#                                                 seq_length=self.seq_length)
#         full_val_dataset = BinaryLogicDataset(seed=self.seed,
#                                               d=self.input_dim,
#                                               batch_size=batch_size,
#                                               seq_length=self.seq_length)
#         full_test_dataset = BinaryLogicDataset(seed=self.seed,
#                                                d=self.input_dim,
#                                                batch_size=batch_size,
#                                                seq_length=self.seq_length)
#         self.train_dataset = full_train_dataset
#         self.val_dataset = full_val_dataset
#         self.test_dataset = full_test_dataset
#
#         # uncomment this to make the computational graph in tensorboard
#         # self.example_input_array = torch.rand((self.seq_length, self.batch_size, self.input_dim + 2))
#
#     def forward(self, x):
#         return self.model(x)
#
#     def on_train_start(self) -> None:
#         print("==================Logging computational graph")
#         self.model.device = self.device
#         self.model.initialize_hidden(self.batch_size, device=self.device)
#
#     def training_step(self, batch, batch_nb):
#         x, y = batch
#
#         x = x.float().squeeze()
#         y = y.float().squeeze()
#
#         temp_x = x[:, 0, :].squeeze().cpu().numpy()
#         temp_y = y[:, 0, :].squeeze().cpu().numpy()
#
#         # plt.subplot(121)
#         # plt.imshow(x[:, 0, :].squeeze().numpy(), cmap='coolwarm')
#         # plt.subplot(122)
#         # plt.imshow(y[:, 0, :].squeeze().numpy(), cmap='coolwarm')
#         #
#         # plt.show()
#
#         self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
#         y_hat = self.forward(x)
#
#         y_hat = torch.stack(y_hat, dim=0).squeeze()
#         y_hat = y_hat.reshape((-1, self.input_dim))
#         y_hat = torch.sigmoid(y_hat)  # constraint the output dimension to [0, 1]
#         # y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
#         # y_hat = y_hat[self.seq_length*2+2:, :, :]
#
#         # y_hat = y_hat.reshape((-1, self.input_dim))
#         # y = y[self.seq_length*2+2:, :, :]
#         y = y.reshape((-1, self.input_dim))
#
#         loss = (y - y_hat)**2  # use the mse loss for
#         loss = torch.sum(loss)
#
#         # compute accuracy on only the y section with output
#         y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
#         y_hat = y_hat[self.seq_length*2+2:, :, :]
#         y_hat = y_hat.reshape((-1, self.input_dim))
#         y = y.reshape((-1, self.batch_size, self.input_dim))
#         y = y[self.seq_length*2+2:, :, :]
#         y = y.reshape((-1, self.input_dim))
#
#         accuracy = ((y_hat[:, InputStream.DATALINE.value] > 0.5).long().flatten() ==
#                     y[:, InputStream.DATALINE.value]).float().mean()
#
#         logs = {"loss": loss}
#         self.log("training loss", loss.cpu().item(), prog_bar=True)
#
#         logs = {"accuracy": accuracy}
#         self.log("accuracy", accuracy.cpu().item(), prog_bar=True)
#
#         logs = {"loss": loss, "accuracy": accuracy}
#         return logs
#
#     def training_end(self, outputs):
#         # print(outputs)
#         outputs["avg_train_accuracy"] = 0
#         return outputs
#
#     def test_step(self, batch, batch_nb):
#         x, y = batch
#         y_hat = self.forward(x)
#         accuracy = (y_hat.argmax(1) == y).float().mean()
#
#         return {'test_accuracy': accuracy}
#
#     def test_end(self, outputs):
#         avg_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
#
#         logs = {"test_accuracy": avg_accuracy}
#
#         return {'test_accuracy': avg_accuracy, "log": logs}
#
#     def configure_optimizers(self):
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return self.optimizer
#
#     def train_dataloader(self):
#         print("-------------------------Initializing train dataloader")
#         return DataLoader(self.train_dataset)
#
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset)
#
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset)
