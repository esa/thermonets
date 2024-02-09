import numpy as np

def rho_approximation(h, params):
    """Approximates rho via the exponenial formula

    Args:
        h (`np.array`): altitude
        params (`np.array`): the [alpha, beta,gamma] parameters

    Returns:
        `np.array`: the density
    """
    n = len(params) // 3
    alphas = params[0:n]
    betas = params[n : 2 * n]
    gammas = params[2 * n : 3 * n]
    retval = np.zeros(h.shape)
    for alpha, beta, gamma in zip(alphas, betas, gammas):
        retval += alpha * np.exp(-(h - gamma) * beta)
    return retval
