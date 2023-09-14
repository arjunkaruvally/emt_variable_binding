import torch


class LSTMModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim)
        self.readout = torch.nn.Linear(hidden_dim, output_dim)
        self.num_layers = 1

    def initialize_hidden(self, batch_size=1, h=None, c=None, device='cpu'):
        if h is None:
            self.h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, requires_grad=True) + \
                     1e-3
            self.c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, requires_grad=True) + \
                     1e-3
        else:
            self.h = h
            self.c = c

    def forward(self, x):
        o, hidden = self.lstm(x, (self.h, self.c))

        self.h, self.c = hidden
        self.all_hidden = [(o[i, :, :], None) for i in range(o.shape[0])]

        all_output = []
        for hidden in self.all_hidden:
            all_output.append(self.readout(hidden[0]))

        return all_output
