import pickle as pk
import torch
import numpy as np

from . import ffnn, normalize_min_max, rho_approximation

# NRLMSISE-00 (loading the ANN and global fit)
with open("../global_fits/global_fit_nrlmsise00_180.0-1000.0-4.txt", "rb") as f:
    best_global_fit_nrlmsise00 = pk.load(f)
model_nrlmsise00 = ffnn(
    input_dim=10,
    hidden_layer_dims=[32, 32],
    output_dim=12,
    mid_activation=torch.nn.Tanh(),
    last_activation=torch.nn.Tanh(),
)
_model_nrlmsise00_path = "../models/nrlmsise00_model_10_32_32_2.60.pyt"
model_nrlmsise00.load_state_dict(torch.load(_model_nrlmsise00_path))

# JB08 (loading the ANN and global fit)
with open("../global_fits/global_fit_jb08_180.0-1000.0-4.txt", "rb") as f:
    best_global_fit_jb08 = pk.load(f)
model_jb08 = ffnn(
    input_dim=19,
    hidden_layer_dims=[32, 32],
    output_dim=12,
    mid_activation=torch.nn.Tanh(),
    last_activation=torch.nn.Tanh(),
)
_model_jb08_path = "../models/jb08_model_10_32_32_1.50.pyt"
model_jb08.load_state_dict(torch.load(_model_jb08_path))


# NRLMSISE-00 (ANN interface)
def nrlmsise00_tn(hs, lons, lats, ap, f107, f107a, doy, sid):
    # We prepare the inputs for the inference
    # lons_, hs_, lats_= np.meshgrid(lons, hs, lats)
    hs_, lons_, lats_ = np.meshgrid(hs, lons, lats, indexing="ij")
    shape = hs_.shape
    size = hs_.size

    # Flatten the grid
    hs_ = hs_.reshape((-1,))
    lons_ = lons_.reshape((-1,))
    lats_ = lats_.reshape((-1,))

    # Prepare the inputs
    nn_in = np.zeros((size, 10))
    nn_in[:, 0] = np.sin(lons_)
    nn_in[:, 1] = np.cos(lons_)
    nn_in[:, 2] = normalize_min_max(lats_, -np.pi / 2, np.pi / 2)
    nn_in[:, 3] = np.sin(2 * np.pi * sid / 86400.0)
    nn_in[:, 4] = np.cos(2 * np.pi * sid / 86400.0)
    nn_in[:, 5] = np.sin(2 * np.pi * doy / 365.25)
    nn_in[:, 6] = np.cos(2 * np.pi * doy / 365.25)
    nn_in[:, 7] = normalize_min_max(f107, 60.0, 266.0)
    nn_in[:, 8] = normalize_min_max(f107a, 60.0, 170.0)
    nn_in[:, 9] = normalize_min_max(ap, 0.0, 110.0)
    nn_in_torch = torch.tensor(nn_in, dtype=torch.float32)

    # Compute the parameters of the exponentials as params_i = params_i0 (1 + NNout_i)
    delta_params = model_nrlmsise00(nn_in_torch)
    params = torch.tensor(best_global_fit_nrlmsise00, dtype=torch.float32) * (
        1 + delta_params
    )
    # Compute the inference of the model over the entire dataset
    predicted = rho_approximation(
        torch.tensor(hs_, dtype=torch.float32), params, backend="torch"
    )
    return predicted.reshape(shape)