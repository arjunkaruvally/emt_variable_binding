import numpy as np

from emt_tools.utils import get_grounded_bases, plot_evolution_in_bases


class LinearModel:
    """
    EMT currently is limited to linear models.
    This class can be used to convert usual non-linear RNNs to an equivalent linear model.
    """

    def __init__(self, task_dimension, hidden_dimension):
        """
        Constructor for the LinearModel class.

        :param task_dimension:
            the dimension of the task space. That is, the dimensions u
            in the task u(t) = f(u(t-1), ... u(t-s))
        :param hidden_dimension:
            dimensions of the hidden state of the RNN
        """
        self.task_dimension = task_dimension
        self.hidden_dimension = hidden_dimension

        self.W_uh = np.zeros((self.task_dimension, self.hidden_dimension))
        self.W_hy = np.zeros((self.hidden_dimension, self.task_dimension))
        self.W_hh = np.zeros((self.hidden_dimension, self.hidden_dimension))

    def parse_simple_rnn(self, model):
        """
        Convert a simple Elman RNN (no bias), with a linear (without bias) output layer to the LinearModel class.

        :param model:
            pytorch rnn model with RNN module named 'rnn' and output layer named 'readout'
        """
        self.W_uh = model.rnn.weight_ih_l0.cpu().detach().numpy()
        self.W_hh = model.rnn.weight_hh_l0.cpu().detach().numpy()
        self.W_hy = model.readout.weight.cpu().detach().numpy()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def parse_lstm(self, model):
        """
        Convert a LSTM (no bias), with a linear (without bias) output layer to the LinearModel class.

        :param model:
            pytorch rnn model with LSTM module named 'lstm' and output layer named 'readout'
        """
        self.W_uh = np.zeros((2 * self.hidden_dimension, self.task_dimension))
        self.W_hh = np.zeros((2 * self.hidden_dimension, 2 * self.hidden_dimension))
        self.W_hy = np.zeros((self.task_dimension, self.hidden_dimension * 2))
        w_hi, w_hf, w_hg, w_ho = np.split(
            model.lstm.weight_hh_l0.cpu().detach().numpy(), 4, 0
        )
        hidden_dim = w_hf.shape[0]

        ## Assemble the jacobian matrix
        self.W_hh[:hidden_dim, hidden_dim:] = np.diag(
            self.sigmoid(np.zeros(hidden_dim))
        )
        self.W_hh[hidden_dim:, :hidden_dim] = (
            np.diag(self.d_sigmoid(np.zeros(hidden_dim))) @ w_hi
            + np.diag(self.sigmoid(np.zeros(hidden_dim))) @ w_hg
        )
        self.W_hh[hidden_dim:, hidden_dim:] = np.diag(
            self.sigmoid(np.zeros(hidden_dim))
        )

        self.W_hy[:, :hidden_dim] = model.readout.weight.cpu().detach().numpy()

    def get_variable_basis(
        self, s, h_simulated, alpha=1, f_operator=None, strength=1, threshold=0.99
    ):
        """
        Get the variable basis for the linear model.
        """
        return get_grounded_bases(
            self.task_dimension,
            self.hidden_dimension,
            self.W_hh,
            self.W_hy,
            s,
            h_simulated=h_simulated,
            alpha=alpha,
            f_operator=f_operator,
            strength=strength,
            threshold=threshold,
        )

    def plot_evolution_basis(
        self,
        u_history,
        h_history,
        y_history,
        Psi_star,
        Phi,
        task_dimension,
        s,
        filename="animation.gif",
    ):
        plot_evolution_in_bases(
            u_history,
            h_history,
            y_history,
            Psi_star,
            Phi,
            task_dimension,
            s,
            filename=filename,
        )
