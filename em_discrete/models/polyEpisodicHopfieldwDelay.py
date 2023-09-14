import torch


class PolyEpisodicRNNModelwDelay(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, poly_power=1):
        super(PolyEpisodicRNNModelwDelay, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.poly_power = poly_power
        self.w_ih = torch.nn.Parameter(torch.zeros((input_dim, hidden_dim)))
        self.w_hh0 = torch.nn.Parameter(torch.zeros((hidden_dim, hidden_dim)))
        self.w_hh1 = torch.nn.Parameter(torch.zeros((hidden_dim, hidden_dim)))
        self.w_ic = torch.nn.Parameter(torch.zeros((input_dim, hidden_dim)))
        self.w_hc = torch.nn.Parameter(torch.zeros((hidden_dim, hidden_dim)))
        self.w_ch = torch.nn.Parameter(torch.zeros((hidden_dim, hidden_dim)))

        self.b1 = torch.nn.Parameter(torch.zeros((hidden_dim, 1)))
        self.b2 = torch.nn.Parameter(torch.zeros((hidden_dim, 1)))
        self.bc = torch.nn.Parameter(torch.zeros((hidden_dim, 1)))

        self.readout = torch.nn.Linear(hidden_dim, output_dim)
        self.num_layers = 1

        # initialization
        # stdv = 1/torch.sqrt(hidden_dim)
        torch.nn.init.xavier_uniform_(self.w_ih)
        torch.nn.init.xavier_uniform_(self.w_hh0)
        torch.nn.init.xavier_uniform_(self.w_hh1)
        torch.nn.init.xavier_uniform_(self.w_ic)
        torch.nn.init.xavier_uniform_(self.w_ch)

        torch.nn.init.xavier_uniform_(self.b1)
        torch.nn.init.xavier_uniform_(self.bc)
        torch.nn.init.xavier_uniform_(self.b2)

    def execute_cell(self, input):
        seq_length, batch_size, d = input.size()
        # output = torch.zeros((seq_length, batch_size, 2 * self.hidden_dim))

        self.all_hidden = []

        for pos_id in range(seq_length):
            # in_arr =
            z = torch.nn.functional.sigmoid((input[pos_id, :, :].squeeze()) @ self.w_ic +
                                            torch.tanh(self.h) @ self.w_hc +
                                            self.bc.view((1, -1)))
            self.c = (1 - z) * torch.tanh(self.h) + z * self.c
            # x = torch.clamp(torch.tanh(self.h) @ self.w_hh0.T + (input[pos_id, :, :].squeeze()) @
            #                           self.w_ih + self.c @ self.w_ch.T + self.b1.T, -1e2, 1e2)
            x = torch.tanh(self.h) @ self.w_hh0.T + (input[pos_id, :, :].squeeze()) @ self.w_ih + self.c @ self.w_ch.T + self.b1.T
            x = torch.pow(x, self.poly_power)
            # x = torch.softmax(x, dim=1)
            #### normalization
            # xsum = torch.sum(x, dim=1)
            # x = x / torch.pow(xsum.view((-1, 1)), 1 / self.poly_power)
            #################
            self.h = x @ self.w_hh1.T + self.b2.T
            # self.h = x
            self.all_hidden.append((self.h, self.c))

    def initialize_hidden(self, batch_size=1, h=None, c=None, device='cpu'):
        if h is None:
            self.h = torch.zeros(batch_size, self.hidden_dim, device=device, requires_grad=True) + \
                     1e-3
            self.c = torch.zeros(batch_size, self.hidden_dim, device=device, requires_grad=True) + \
                     1e-3
        else:
            self.h = h
            self.c = c

    def forward(self, x):
        self.execute_cell(x)
        # reshaped_hidden = o.view((-1, 2*self.hidden_dim))[:, :self.hidden_dim]
        all_output = []

        for hidden, c in self.all_hidden:
            all_output.append(self.readout(hidden))

        return all_output
