import os

import numpy as np

from pleque.io import _geqdsk
from pleque.io.geqdsk import read
from pleque.io.readers import read_geqdsk


def test_calculated_profiles(geqdsk_file):
    """
    Octants: https://en.wikipedia.org/wiki/Octant_(solid_geometry)#/media/File:Octant_numbers.svg

    :param geqdsk_file:
    :return:
    """

    eq = read_geqdsk(geqdsk_file, cocos=3)
    with open(geqdsk_file, 'r') as f:
        eq_dict = _geqdsk.read(f)

    pressure = eq_dict['pres']
    pprime = eq_dict['pprime']

    F = eq_dict['F']
    FFprime = eq_dict['FFprime']

    psi_n = np.linspace(0, 1, len(pressure))

    assert np.allclose(pressure, eq.pressure(psi_n=psi_n))
    assert np.allclose(pprime, eq.pprime(psi_n=psi_n), atol=100)

    assert np.allclose(F, eq.F(psi_n=psi_n), atol=1e-2)
    assert np.allclose(FFprime, eq.FFprime(psi_n=psi_n), atol=1e-3)



def test_from_to_gfile(equilibrium):
    file_name = '/tmp/g{:d}.{:.0f}{}'.format(equilibrium.shot, int(equilibrium.time), equilibrium.time_unit)

    equilibrium.to_geqdsk(file_name)

    eq2 = read(file_name)
    os.remove(file_name)

    assert np.isclose(equilibrium.magnetic_axis.R, eq2.magnetic_axis.R, atol=1e-5, rtol=1e-4)
    assert np.isclose(equilibrium.magnetic_axis.Z, eq2.magnetic_axis.Z, atol=1e-5, rtol=1e-4)





