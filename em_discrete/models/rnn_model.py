import torch


class RNNModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, bias=False):
        super(RNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, nonlinearity='tanh', bias=bias)
        self.readout = torch.nn.Linear(hidden_dim, output_dim, bias=bias)
        self.num_layers = 1

    def initialize_hidden(self, batch_size=1, h=None, device='cpu'):
        if h is None:
            self.h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, requires_grad=True)
        else:
            self.h = h

    def forward(self, x):
        o, self.h = self.rnn(x, self.h)
        # print(x.shape, o.shape)
        self.all_hidden = o.detach().clone()
        # self.all_hidden = [(o[i, :, :], None) for i in range(o.shape[0])]
        #
        # all_output = []
        # for hidden in self.all_hidden:
        #     all_output.append(self.readout(hidden[0]))

        all_output = self.readout(o)

        return all_output
