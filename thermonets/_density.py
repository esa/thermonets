import numpy as np
import pygmo as pg
import torch

def rho_approximation(h, params,backend='numpy'):
    """Approximates rho via the exponenial formula

    Args:
        h (`np.array` or `torch.tensor`): altitude
        params (`np.array` or `torch.tensor`): the [alpha, beta,gamma] parameters

    Returns:
        `np.array` or `torch.tensors`: the density
    """        
    if backend == 'numpy':      
        n = len(params) // 3
        alphas = params[0:n]
        betas = params[n : 2 * n]
        gammas = params[2 * n : 3 * n]
        retval = np.zeros(h.shape)
        for alpha, beta, gamma in zip(alphas, betas, gammas):
            retval += alpha * np.exp(-(h - gamma) * beta)
    elif backend=='torch':
        n = params.shape[1] // 3
        alphas = params[:,0:n]
        betas = params[:,n : 2 * n]
        gammas = params[:,2 * n : 3 * n]
        retval = torch.zeros_like(h)
        for i in range(n):
            retval += alphas[:,i] * torch.exp(-(h - gammas[:,i]) * betas[:,i])
    return retval

class global_fit_udp:
    """This UDP will find a fit for h,rho data to the shape a_i*exp(-b_i*(h-g_i)), i=1..n
    
    The UDP is not agnostic to the units, as the problem bounds have been tuned
    so that h is expected to be in km and rho in kg/m^3.

    """
    def __init__(self, X, Y, n=4):
        self.X = X
        self.Y = Y
        self.n = n

    def fitness(self,x):
        Y_pred = rho_approximation(self.X, x)
        return [np.mean(np.abs(np.log10(self.Y)-np.log10(Y_pred)))]

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

    def get_bounds(self):
        lb = [0]*self.n*3
        ub = [1e-1]*self.n*3
        ub[self.n*2:] = [100]*self.n
        return (lb, ub)
