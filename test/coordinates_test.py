from pleque.coordinates import Coordinates
from test.testing_utils import load_testing_equilibrium
import numpy as np

eq = load_testing_equilibrium()

def test_coords_2d(*coordinates, R = None, Z = None, coord_type=None, grid=False, **coords):

    print('--------')
    xy = Coordinates(eq, *coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
    print('dim = {}'.format(xy.dim))
    print('x1 = {}'.format(xy.x1))
    print('x2 = {}'.format(xy.x2))
    print('coord_type = {}'.format(xy.coord_type))
    assert xy.dim == 2
    assert isinstance(xy.x1, np.ndarray)
    assert isinstance(xy.x2, np.ndarray)
    print('--------')
    print()
    return xy

def test_coords_1d(*coordinates, psi_n = None, coord_type=None, grid=False, **coords):

    xy = Coordinates(eq, *coordinates, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
    print('--------')
    print('dim = {}'.format(xy.dim))
    print('x1 = {}'.format(xy.x1))
    print('coord_type = {}'.format(xy.coord_type))
    assert xy.dim == 1
    assert isinstance(xy.x1, np.ndarray)
    print('--------')
    print()
    return xy

if __name__ == '__main__':

    # 2d tests (R, Z)

    coord = test_coords_2d(1, 2)
    assert coord.coord_type == ('R', 'Z')
    coord = test_coords_2d(R=1, Z=2)
    assert coord.coord_type == ('R', 'Z')
    coord = test_coords_2d([1, 2, 3, 4], [3, 5, 6, 2])
    assert coord.coord_type == ('R', 'Z')
    coord = test_coords_2d(([1, 3], [3, 5], [5, 3]))
    assert coord.coord_type == ('R', 'Z')
    coord = test_coords_2d(np.array(([1, 3], [3, 5], [5, 3])), coord_type=['Z', 'R'])
    assert coord.coord_type == ('R', 'Z')
    coord = test_coords_2d(R=[1, 2, 3, 4], Z=[3, 5, 6, 2])
    assert coord.coord_type == ('R', 'Z')
    coord = test_coords_2d(Z=[3, 5, 6, 2], R=[1, 2, 3, 4])
    assert coord.coord_type == ('R', 'Z')

    # 1d tests (psi_n)
    coord = test_coords_1d(0.5)
    assert coord.coord_type == ('psi_n',)
    coord = test_coords_1d(psi_n=0.5)
    assert coord.coord_type == ('psi_n',)
    coord = test_coords_1d(np.linspace(0, 1, 6))
    assert coord.coord_type == ('psi_n',)
    coord = test_coords_1d(psi=[0,0.2,0.4,0.6,0.8,1])
    assert coord.coord_type == ('psi',)
    coord = test_coords_1d(rho=[0, 0.2, 0.4, 0.6, 0.8, 1])
    assert coord.coord_type == ('psi',)
    coord = test_coords_1d(np.linspace(0, 1, 6), coord_type=('rho',))
    assert coord.coord_type == ('psi_n',)

    # 0d case
    xy = Coordinates(eq)
    assert xy.dim == 0