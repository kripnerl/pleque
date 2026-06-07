import numpy as np

from pleque.core import Coordinates
from pleque.tests.utils import load_testing_equilibrium

# todo: rewrite test to not use this
eq = load_testing_equilibrium()


def coords_3d(*coordinates, coord_type=None, **coords):
    xy = eq.coordinates(*coordinates, coord_type=coord_type, grid=False, **coords)
    assert xy.dim == 3
    assert isinstance(xy._x1_input, np.ndarray)
    assert isinstance(xy._x2_input, np.ndarray)
    return xy


def coords_2d(*coordinates, R=None, Z=None, coord_type=None, grid=False, **coords):
    print('--------')
    xy = eq.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
    print(f'_coord_type_input = {xy._coord_type_input}')
    assert xy.dim == 2
    assert isinstance(xy._x1_input, np.ndarray)
    assert isinstance(xy._x2_input, np.ndarray)
    print('--------')
    print()
    return xy


def coords_1d(*coordinates, psi_n=None, coord_type=None, grid=False, **coords):
    xy = eq.coordinates(*coordinates, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)

    print('--------')
    print(f'dim = {xy.dim}')
    print(f'_x1_input = {xy._x1_input}')
    print(f'_coord_type_input = {xy._coord_type_input}')
    assert xy.dim == 1
    assert isinstance(xy._x1_input, np.ndarray)
    print('--------')
    print()
    return xy


def compare_arrays(a1, a2):
    assert len(a1) == len(a2)
    for i in np.arange(len(a1)):
        assert np.abs(a1[i] - a2[i]) < 1e-3

def test_midplane(equilibrium):

    axis = equilibrium.coordinates(psi_n = 0)
    lcfs = equilibrium.coordinates(psi_n = 1)

    np.testing.assert_almost_equal(axis.as_RZ_mid().R, equilibrium._mg_axis[0])
    np.testing.assert_almost_equal(axis.as_RZ_mid().Z, equilibrium._mg_axis[1])
    np.testing.assert_almost_equal(equilibrium._mg_axis[0] + lcfs.r_mid, lcfs.R_mid)

def test_scalar_point_coordinate(equilibrium):

    coord = Coordinates(None, R=1, Z=2)
    assert len(coord) == 1
    assert coord.x1[0] == 1
    assert coord.x2[0] == 2

    coord = Coordinates(None, R=1, Z=2, phi=0)
    assert len(coord) == 1
    assert coord.x1[0] == 1
    assert coord.x2[0] == 2

    coord = equilibrium.coordinates(R=1, Z=2)
    assert len(coord) == 1
    assert coord.x1[0] == 1
    assert coord.x2[0] == 2

def test_coordinates(equilibrium):
    # coord = eq.coordinates(eq._lcfs)

    # 2d tests (R, Z)
    #
    coord = coords_2d(1, 2)
    assert coord._coord_type_input == ('R', 'Z')
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_2d(R=1, Z=2)
    assert coord._coord_type_input == ('R', 'Z')
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_2d([1, 2, 3, 4], [3, 5, 6, 2])
    compare_arrays([1, 2, 3, 4], coord.R)
    assert coord._coord_type_input == ('R', 'Z')
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_2d(([1, 3], [3, 5], [5, 3]))
    assert coord._coord_type_input == ('R', 'Z')
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_2d(np.array(([1, 3], [3, 5], [5, 3])), coord_type=['Z', 'R'])
    # compare_arrays([1, 3, 5], coord.Z)
    # compare_arrays([3, 5, 3], coord.R)
    assert coord._coord_type_input == ('R', 'Z')
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_2d(R=[1, 2, 3, 4], Z=[3, 5, 6, 2])
    compare_arrays([3, 5, 6, 2], coord.Z)
    assert coord._coord_type_input == ('R', 'Z')
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_2d(Z=[3, 5, 6, 2], R=[1, 2, 3, 4])
    assert coord._coord_type_input == ('R', 'Z')
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_2d(r=[0.2, 0.3, 0.3, 0.2], theta=[0, np.pi / 2, np.pi, 3 / 2 * np.pi])
    compare_arrays([0, np.pi / 2, np.pi, -np.pi / 2], coord.theta)
    compare_arrays([0.2, 0.3, 0.3, 0.2], coord.r)
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    # 1d tests (psi_n)
    coord = coords_1d(0.5)
    assert coord._coord_type_input == ('psi_n',)
    assert len(coord.x1.shape) == 1

    coord = coords_1d(psi_n=0.5)
    assert coord._coord_type_input == ('psi_n',)
    assert len(coord.x1.shape) == 1
    compare_arrays([0.5], coord.psi_n)

    coord = coords_1d(np.linspace(0, 1, 6), coord_type='rho')
    compare_arrays(np.linspace(0, 1, 6), coord.rho)
    assert coord._coord_type_input == ('rho',)
    assert len(coord.x1.shape) == 1

    coord = coords_1d(psi=[0.4, 0.35, 0.3, 0.2, 0.15])
    compare_arrays([0.4, 0.35, 0.3, 0.2, 0.15], coord.psi)
    assert coord._coord_type_input == ('psi',)
    assert len(coord.x1.shape) == 1

    coord = coords_1d(rho=[0, 0.2, 0.4, 0.6, 0.8, 1])
    assert coord._coord_type_input == ('rho',)
    assert len(coord.x1.shape) == 1

    coord = coords_1d(np.linspace(0, 1, 6), coord_type=('rho',))
    assert coord._coord_type_input == ('rho',)
    assert len(coord.x1.shape) == 1

    coord2 = eq.coordinates(coord)
    assert coord is coord2

    coord = eq.coordinates(psi_n=np.linspace(0, 1, 10))
    print(f'r_mid = {coord.r_mid}')

    coord = eq.coordinates(eq._mg_axis[0], eq._mg_axis[1])
    compare_arrays(coord.psi_n, [0])

    # 3d case:
    coord = coords_3d(np.linspace(1, 5, 11), np.zeros(11), np.zeros(11))
    compare_arrays(coord.X, coord.R)
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_3d(X=np.linspace(1, 5, 11), Y=np.zeros(11), Z=np.zeros(11))
    compare_arrays(coord.X, coord.R)
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    coord = coords_3d(np.linspace(1, 5, 11), np.zeros(11), np.ones(11) * np.pi / 2)
    compare_arrays(coord.Y, coord.R)
    assert len(coord.x1.shape) == 1
    assert len(coord.x2.shape) == 1

    # 0d case
    xy = Coordinates(eq)
    assert xy.dim == 0

def test_intersections(equilibrium):
    # One point
    coord1 = equilibrium.coordinates(R=[-1, 1], Z=[0, 0])
    coord2 = equilibrium.coordinates(R=[0, 0], Z=[-1, 1])

    intersections = coord1.intersection(coord2)

    assert len(intersections) == 1
    assert np.isclose(intersections.R[0], 0)
    assert np.isclose(intersections.Z[0], 0)

    # Two points
    coord1 = equilibrium.coordinates(R=[-1, 1], Z=[0, 0])
    coord2 = equilibrium.coordinates(R=[-1, 0, 1], Z=[-1, 1, -1])
    intersections = coord1.intersection(coord2)

    assert len(intersections) == 2
    assert np.isclose(intersections.R[0], -0.5) or np.isclose(intersections.R[0], 0.5)
    assert np.isclose(intersections.Z[0], 0)

    # No intersection
    coord1 = equilibrium.coordinates(R=[-1, 1], Z=[0, 0])
    coord2 = equilibrium.coordinates(R=[-1, 1], Z=[1, 1])
    intersections = coord1.intersection(coord2)

    assert intersections is None


def test_array_input():

    r = np.linspace(1, 2, 10)
    z = np.linspace(-1, 1, 12)

    rr, zz = np.meshgrid(r, z)

    coord = Coordinates(eq, R=rr, Z=zz, grid=False)

    assert coord.R.shape == rr.shape
    assert coord.Z.shape == rr.shape
    assert coord.r_mid.shape == rr.shape

    coord = Coordinates(eq, r=coord.r_mid, theta=np.zeros_like(coord.r_mid), grid=False)

    assert coord.R.shape == rr.shape
    assert coord.Z.shape == rr.shape
    assert coord.r_mid.shape == rr.shape

    B_midplane = eq.B_pol(r=coord.r_mid, theta=np.zeros_like(coord.r_mid), grid=False)
    B_coords = eq.B_pol(coord)

    assert B_midplane.shape == rr.shape
    assert B_coords.shape == rr.shape
    assert B_midplane.shape == B_coords.shape



def test_distances():

        R = 2
        N = 52

        coord = Coordinates(None, R=np.ones(N) * R, Z=np.zeros(N), phi=np.linspace(0, 2 * np.pi, N))
        calc_length = R * np.pi * 2

        assert np.isclose(coord.length, calc_length, atol=1e-2, rtol=1e-2)
