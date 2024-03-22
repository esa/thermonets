import pickle as pk
import torch
import numpy as np

from . import ffnn, ffnn_numpy, normalize_min_max, rho_approximation

#NRLMSISE-00 (loading the weights and biases of the FFNN and global fit):
with open("../global_fits/global_fit_nrlmsise00_180.0-1000.0-4.txt", "rb") as f:
    best_global_fit_nrlmsise00 = pk.load(f)
_model_nrlmsise00_path='../models/nn_parameters_nrlmsise00_model_10_32_32_2.17.pk'
with open(_model_nrlmsise00_path,'rb') as f:
    (weights_nrlmsise00,biases_nrlmsise00)=pk.load(f)
model_nrlmsise00 = ffnn_numpy(weights=weights_nrlmsise00,
                              biases=biases_nrlmsise00,
                              mid_activation=np.tanh,
                              last_activation=np.tanh)

#JB-08 (loading the weights and biases of the FFNN and global fit):
with open("../global_fits/global_fit_jb08_180.0-1000.0-4.txt", "rb") as f:
    best_global_fit_jb08 = pk.load(f)
_model_jb08_path='../models/nn_parameters_jb08_model_10_32_32_1.43.pk'
with open(_model_jb08_path,'rb') as f:
    (weights_jb08,biases_jb08)=pk.load(f)
model_jb08 = ffnn_numpy(weights=weights_jb08,
                              biases=biases_jb08,
                              mid_activation=np.tanh,
                              last_activation=np.tanh)

# NRLMSISE-00 (ANN interface)
def nrlmsise00_tn(hs, lons, lats, f107a, f107, ap, doy, sid):
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

    # Compute the parameters of the exponentials as params_i = params_i0 (1 + NNout_i)
    delta_params = model_nrlmsise00(nn_in)
    # we extend the gamma parameter as we did in training (otherwise they were saturating)
    delta_params[:,8:]=delta_params[:,8:]*5.
    params = best_global_fit_nrlmsise00 * (1 + delta_params)
    # Compute the inference of the model over the entire dataset
    predicted = rho_approximation(hs_, params, backend="numpy")
    return predicted.reshape(shape)

# JB-08 (ANN interface)
def jb08_tn(hs, lons, lats, f107, f107a, s107, s107a, m107, m107a, y107, y107a, dDstdT, doy, sid):
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
    nn_in = np.zeros((size, 16))
    nn_in[:, 0] = np.sin(lons_)
    nn_in[:, 1] = np.cos(lons_)
    nn_in[:, 2] = normalize_min_max(lats_, -np.pi / 2, np.pi / 2)
    nn_in[:, 3] = np.sin(2 * np.pi * sid / 86400.0)
    nn_in[:, 4] = np.cos(2 * np.pi * sid / 86400.0)
    nn_in[:, 5] = np.sin(2 * np.pi * doy / 365.25)
    nn_in[:, 6] = np.cos(2 * np.pi * doy / 365.25)
    nn_in[:, 7] = normalize_min_max(f107, 60.0, 266.0)
    nn_in[:, 8] = normalize_min_max(f107a, 60.0, 170.0)
    nn_in[:, 9] = normalize_min_max(s107, 50.0, 190.0)
    nn_in[:, 10] = normalize_min_max(s107a, 50.0, 170.0)
    nn_in[:, 11] = normalize_min_max(m107, 50.0, 190.0)
    nn_in[:, 12] = normalize_min_max(m107a, 50.0, 160.0)
    nn_in[:, 13] = normalize_min_max(y107, 50.0, 180.0)
    nn_in[:, 14] = normalize_min_max(y107a, 50.0, 170.0)
    nn_in[:, 15] = normalize_min_max(dDstdT, 0.0, 390.0)

    # Compute the parameters of the exponentials as params_i = params_i0 (1 + NNout_i)
    delta_params = model_jb08(nn_in)
    params = best_global_fit_jb08 * (1 + delta_params)
    # Compute the inference of the model over the entire dataset
    predicted = rho_approximation(hs_, params, backend="numpy")
    return predicted.reshape(shape)