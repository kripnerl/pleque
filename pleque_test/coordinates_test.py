import numpy as np

from pleque.core import Coordinates
from pleque_test.testing_utils import load_testing_equilibrium

eq = load_testing_equilibrium()


def test_coords_3d(*coordinates, coord_type=None, **coords):
    xy = eq.coordinates(*coordinates, coord_type=coord_type, grid=False, **coords)
    assert xy.dim == 3
    assert isinstance(xy._x1_input, np.ndarray)
    assert isinstance(xy._x2_input, np.ndarray)
    return xy


def test_coords_2d(*coordinates, R=None, Z=None, coord_type=None, grid=False, **coords):
    print('--------')
    xy = eq.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
    print('_coord_type_input = {}'.format(xy._coord_type_input))
    assert xy.dim == 2
    assert isinstance(xy._x1_input, np.ndarray)
    assert isinstance(xy._x2_input, np.ndarray)
    print('--------')
    print()
    return xy


def test_coords_1d(*coordinates, psi_n=None, coord_type=None, grid=False, **coords):
    xy = eq.coordinates(*coordinates, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)

    print('--------')
    print('dim = {}'.format(xy.dim))
    print('_x1_input = {}'.format(xy._x1_input))
    print('_coord_type_input = {}'.format(xy._coord_type_input))
    assert xy.dim == 1
    assert isinstance(xy._x1_input, np.ndarray)
    print('--------')
    print()
    return xy


def test_arrays(a1, a2):
    assert len(a1) == len(a2)
    for i in np.arange(len(a1)):
        assert np.abs(a1[i] - a2[i]) < 1e-3


if __name__ == '__main__':
    # coord = eq.coordinates(eq._lcfs)

    # 2d tests (R, Z)
    #
    coord = test_coords_2d(1, 2)
    assert coord._coord_type_input == ('R', 'Z')
    coord = test_coords_2d(R=1, Z=2)
    assert coord._coord_type_input == ('R', 'Z')
    coord = test_coords_2d([1, 2, 3, 4], [3, 5, 6, 2])
    test_arrays([1, 2, 3, 4], coord.R)
    assert coord._coord_type_input == ('R', 'Z')
    coord = test_coords_2d(([1, 3], [3, 5], [5, 3]))
    assert coord._coord_type_input == ('R', 'Z')
    coord = test_coords_2d(np.array(([1, 3], [3, 5], [5, 3])), coord_type=['Z', 'R'])
    # test_arrays([1, 3, 5], coord.Z)
    # test_arrays([3, 5, 3], coord.R)
    assert coord._coord_type_input == ('R', 'Z')
    coord = test_coords_2d(R=[1, 2, 3, 4], Z=[3, 5, 6, 2])
    test_arrays([3, 5, 6, 2], coord.Z)
    assert coord._coord_type_input == ('R', 'Z')
    coord = test_coords_2d(Z=[3, 5, 6, 2], R=[1, 2, 3, 4])
    assert coord._coord_type_input == ('R', 'Z')

    coord = test_coords_2d(r=[0.2, 0.3, 0.3, 0.2], theta=[0, np.pi / 2, np.pi, 3 / 2 * np.pi])
    test_arrays([0, np.pi / 2, np.pi, -np.pi / 2], coord.theta)
    test_arrays([0.2, 0.3, 0.3, 0.2], coord.r)

    # 1d tests (psi_n)
    coord = test_coords_1d(0.5)
    assert coord._coord_type_input == ('psi_n',)

    coord = test_coords_1d(psi_n=0.5)
    assert coord._coord_type_input == ('psi_n',)
    test_arrays([0.5], coord.psi_n)

    coord = test_coords_1d(np.linspace(0, 1, 6), coord_type='rho')
    test_arrays(np.linspace(0, 1, 6), coord.rho)
    assert coord._coord_type_input == ('rho',)

    coord = test_coords_1d(psi=[0.4, 0.35, 0.3, 0.2, 0.15])
    test_arrays([0.4, 0.35, 0.3, 0.2, 0.15], coord.psi)
    assert coord._coord_type_input == ('psi',)

    coord = test_coords_1d(rho=[0, 0.2, 0.4, 0.6, 0.8, 1])
    assert coord._coord_type_input == ('rho',)

    coord = test_coords_1d(np.linspace(0, 1, 6), coord_type=('rho',))
    assert coord._coord_type_input == ('rho',)

    coord2 = eq.coordinates(coord)
    assert coord is coord2

    coord = eq.coordinates(psi_n = np.linspace(0, 1, 10))
    print('r_mid = {}'.format(coord.r_mid))

    coord = eq.coordinates(eq._mg_axis[0], eq._mg_axis[1])
    test_arrays(coord.psi_n, [0])

    # 3d case:
    coord = test_coords_3d(np.linspace(1, 5, 11), np.zeros(11), np.zeros(11))
    test_arrays(coord.X, coord.R)

    coord = test_coords_3d(X=np.linspace(1, 5, 11), Y=np.zeros(11), Z=np.zeros(11))
    test_arrays(coord.X, coord.R)

    coord = test_coords_3d(np.linspace(1, 5, 11), np.zeros(11), np.ones(11)*np.pi/2)
    test_arrays(coord.Y, coord.R)

    # 0d case
    xy = Coordinates(eq)
    assert xy.dim == 0