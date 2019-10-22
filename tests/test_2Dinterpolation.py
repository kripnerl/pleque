import numpy as np
from pleque.tests.utils import load_testing_equilibrium
import pytest


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.
    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


# test of 2D gaussian function
N = 100
z = np.linspace(-0.1, 0.4, N)
r = np.linspace(0.3, 0.8, N)
R, Z = np.meshgrid(r, z)

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[1., -0.5], [-0.5, 1.5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(R.shape + (2,))
pos[:, :, 0] = R
pos[:, :, 1] = Z

# The distribution on the variables X, Y packed into pos.
F1 = multivariate_gaussian(pos, mu, Sigma)
# random function
F2 = np.exp(-R/10.) * np.exp(-((Z-2)/1.5)**3)

eq = load_testing_equilibrium()
@pytest.mark.parametrize("data, coord1, coord2, spline_order, spline_smooth", [(F1, r, z, 3, 1), (F2, r, z, 3, 0)])
def test_surfacefunction(equilibrium, data, coord1, coord2, spline_order, spline_smooth):
    """
    :param eq:
    :param coord1: first coordinate in 2D space
    :param coord2: second coordinate in 2D space
    :param data: function value for fiven coordinates
    :return: 2D spline
    """
    results2 = equilibrium.surfacefuncs.add_surface_func('test', data, coord1, coord2, spline_order=spline_order,
                                                          spline_smooth=spline_smooth)
    delta = results2(coord1, coord2) - data
    print('average error between original values and 2D spline is {}:'.format(float(np.mean(delta))))





