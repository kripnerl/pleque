from pleque.tests.utils import get_test_cases_number, get_test_equilibria_filenames
from pleque.io.readers import read_geqdsk
from pleque import Coordinates
from numpy import array
import numpy as np

import pytest


# import matplotlib.pyplot as plt
#
# from pleque.utils.plotting import plot_extremes, _plot_debug


def test_critical(equilibrium):
    """
    Test if all critical points are properly set. It differ for x-point and limter plasma.
    """

    # # x-point plasma:
    # This code will be added soon
    # if equilibrium.is_xpoint_plasma:
    #     assert equilibrium.x_point == equilibrium.limiter_point
    #     assert np.isclose(equilibrium._psi_lcfs, equilibrium._psi_xp)
    #     assert equilibrium.contact_point is None
    #     if len(equilibrium.first_wall) > 4:
    #         assert len(equilibrium.strike_points) > 1
    #
    # else:
    #     assert equilibrium.contact_point == equilibrium.strike_points
    #     assert equilibrium.contact_point == equilibrium.limiter_point
    pass


o_points = [array([0.90891258, 0.01440663]), array([0.9083923, 0.06746012]),
            array([9.10399169e-01, 2.64745319e-07]), array([0.56788997, 0.00524001]),
            array([0.55436531, 0.02267376]), array([0.56631465, 0.01856815])]
x_points = [array([0.74987394, -0.49864919]), array([0.74987669, -0.44865779]),
            array([0.73877241, 0.49947451]), None,
            array([0.47302589, -0.33277101]), array([0.46132649, -0.33223827])]
st_points = [None, None, None, array([0.347, 0.00733336]), None, None]


@pytest.mark.parametrize(('case',), [[0], [1], [2], [3], [4], [5]])
def test_equilibria(case):

    gfiles = get_test_equilibria_filenames()

    print("Reading {}".format(gfiles[case]))

    eq = read_geqdsk(gfiles[case])

    assert np.allclose(eq._mg_axis, o_points[case])
    if eq._x_point is not None:
        assert np.allclose(eq._x_point, x_points[case])
    if eq._strike_points is not None and st_points[case] is not None:
        assert np.allclose(eq._strike_points[0], st_points[case])


def test_eq_properties(equilibrium):
    print(equilibrium.first_wall.R[0])
    assert np.isclose(equilibrium.magnetic_axis.psi_n, 0)
    if len(equilibrium.first_wall) > 4:
        assert isinstance(equilibrium.strike_points, Coordinates)
    else:
        assert equilibrium._strike_points is None

    if equilibrium._limiter_plasma:
        assert isinstance(equilibrium.contact_point, Coordinates)
        assert isinstance(equilibrium.strike_points, Coordinates)
    else:
        assert equilibrium.contact_point is None
        if len(equilibrium.first_wall) > 4:
            assert len(equilibrium.strike_points) > 1

    assert equilibrium.B_R(R=np.linspace(0.3, 0.5, 10), Z=np.zeros(15),
                           grid=True).shape == (15, 10)

    assert equilibrium.Bvec(R=np.linspace(0.3, 0.5, 10), Z=np.zeros(10),
                            swap_order=False, grid=False).shape == (3, 10)

    assert equilibrium.Bvec(R=np.linspace(0.3, 0.5, 10), Z=np.zeros(10),
                            swap_order=True, grid=False).shape == (10, 3)

    assert equilibrium.Bvec(R=np.linspace(0.3, 0.5, 10), Z=np.zeros(15),
                            swap_order=False, grid=True).shape == (3, 15, 10)

    assert equilibrium.Bvec(R=np.linspace(0.3, 0.5, 10), Z=np.zeros(15),
                            swap_order=True, grid=True).shape == (15, 10, 3)

    assert equilibrium.Bvec_norm(R=np.linspace(0.3, 0.5, 10), Z=np.zeros(15),
                                 swap_order=False, grid=True).shape == (3, 15, 10)

    assert equilibrium.Bvec_norm(R=np.linspace(0.3, 0.5, 10), Z=np.zeros(15),
                                 swap_order=True, grid=True).shape == (15, 10, 3)