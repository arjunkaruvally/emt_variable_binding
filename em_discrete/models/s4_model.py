import sys

import torch
import torch.nn.functional as F
from em_discrete.models.S4_layer import S4Layer

class S4Model(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, l_max=200, bias=False):
        super(S4Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = l_max
        self.rnn = S4Layer(input_dim, hidden_dim, l_max)
        self.readout = torch.nn.Linear(hidden_dim, output_dim, bias=bias)
        self.num_layers = 1

    def initialize_hidden(self, batch_size=1, h=None, device='cpu'):
        if h is None:
            self.h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, requires_grad=True)
        else:
            self.h = h

    def forward(self, x):

        x = torch.swapaxes(x, 0, 1)
        input_seq_length = x.shape[1]
        x = F.pad(x, (0, 0, 0, self.max_seq_length - input_seq_length, 0, 0), "constant", 0)
        h = self.rnn(x)
        h = h[:, :input_seq_length, :]
        h = torch.swapaxes(h, 0, 1)

        # print(h.shape)
        return h
