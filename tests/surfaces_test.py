# import matplotlib.pyplot as plt
import numpy as np

from pleque.utils.surfaces import find_contour, get_surface, point_in_first_wall, points_inside_curve
from pleque.tests.utils import load_testing_equilibrium
import pleque.utils.surfaces as surf


def test_surfs():
    # test intersection function:

    l1 = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
    l2 = np.array([[-2, 0], [2, 0]])
    l3 = np.array([[-2, 0], [0, 0]])
    l4 = np.array([[10, 10], [20, 10]])

    i1 = surf.intersection(l1, l2)
    i2 = surf.intersection(l1, l3)
    i3 = surf.intersection(l1, l4)

    assert np.shape(i1) == (2, 2)
    assert np.shape(i2) == (1, 2)
    assert i3 is None

    assert np.isclose(i2[0, 0], -1)
