import torch
from torch import nn
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

def MSE():
    return torch.nn.MSELoss()

def MSE_LOG10():
    return lambda x,y: torch.nn.MSELoss()(torch.log10(x), torch.log10(y))

def MAPE():
    return lambda x,y: mean_absolute_percentage_error(x,y)

