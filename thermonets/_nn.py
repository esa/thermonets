import torch
from torch import nn
import numpy as np
import pickle
from ._util import mean_absolute_percentage_error

class ffnn(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_layer_dims=[32, 32],
        output_dim=2,
        mid_activation=nn.Tanh(),
        last_activation=nn.Tanh()
    ):
        super(ffnn, self).__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.act = mid_activation
        self.acts = nn.ModuleList([self.act for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)
        self.last_activation = last_activation

    def forward(self, x):
        for fc, act in zip(self.fcs, self.acts):
            x = act(fc(x))
        return self.last_activation(self.fc_out(x))

class ffnn_numpy:
    def __init__(
            self,
            weights,
            biases,
            mid_activation=np.tanh,
            last_activation=np.tanh
    ):
        """
        Args:
            - weights: list of numpy arrays
            - biases: list of numpy arrays
            - mid_activation: activation function of all layers except last
            - last_activation: activation function of last layers
        """
        self.weights = weights
        self.biases = biases
        self.mid_activation=mid_activation
        self.last_activation=last_activation

    def __call__(self,x):
        for i in range(len(self.weights)-1):
            x = self.mid_activation(x @ self.weights[i] + self.biases[i])
        return self.last_activation( x @ self.weights[-1]+ self.biases[-1])

def MSE():
    return torch.nn.MSELoss()

def MSE_LOG10():
    return lambda x,y: torch.nn.MSELoss()(torch.log10(x), torch.log10(y))

def MAPE():
    return lambda x,y: mean_absolute_percentage_error(x,y)

