import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pleque.tests.utils as test_util
from pleque.core import SurfaceFunctions


def test_surfacefunction(equilibrium, coord1, coord2, data, spline_order=3, spline_smooth=1):
    """
    :param eq:
    :param coord1: first coordinate in 2D space
    :param coord2: second coordinate in 2D space
    :param data: function value for fiven coordinates
    :return: 2D spline
    """
    vysledok2 = equilibrium.surfacefuncs.add_surface_func('test', data, coord1, coord2, spline_order=spline_order,
                                                 spline_smooth=spline_smooth)
    return vysledok2


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



if test_util.get_test_cases_number():
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
    F = multivariate_gaussian(pos, mu, Sigma)
    
    eq = test_util.load_testing_equilibrium(1)
    spline2d = test_surfacefunction(eq, r, z, F, spline_order=3, spline_smooth=1)

    fig, ax = plt.subplots()
    ax.contour(R, Z, F, 30, cmap=cm.viridis)
    ax.contour(r, z, spline2d(r, z), 30, colors='k', linestyles=':')
    plt.show()

else:
    print('testing equilibrium was not found')

