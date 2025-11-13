import torch


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=False):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, bias=bias)
        self.readout = torch.nn.Linear(hidden_dim, output_dim, bias=bias)
        self.num_layers = 1

    def initialize_hidden(self, batch_size=1, h=None, c=None, device="cpu"):
        if h is None:
            self.h = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=device,
                requires_grad=True,
            )
            self.c = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=device,
                requires_grad=True,
            )
        else:
            self.h = h
            self.c = c

    def forward(self, x):
        o, hidden = self.lstm(x, (self.h, self.c))

        self.h, self.c = hidden
        self.all_hidden = o.detach().clone()

        all_output = self.readout(o)

        return all_output
