import os
import tempfile

import numpy as np

import pleque
from pleque.io import _geqdsk
from pleque.io.geqdsk import read
from pleque.io.readers import read_geqdsk

import pkg_resources
import xarray as xr


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

    assert np.allclose(pressure, eq.pressure(psi_n=psi_n), atol=1)
    assert np.allclose(pprime, eq.pprime(psi_n=psi_n), atopkg_resourcesl=100)

    assert np.allclose(F, eq.F(psi_n=psi_n), atol=1e-2)
    assert np.allclose(FFprime, eq.FFprime(psi_n=psi_n), atol=1e-3)



def test_from_to_gfile(equilibrium):
    tmp_dir = tempfile.TemporaryDirectory()
    file_name = '{}/g{:d}.{:.0f}{}'.format(tmp_dir.name, equilibrium.shot, int(equilibrium.time), equilibrium.time_unit)

    print(file_name)

    equilibrium.to_geqdsk(file_name)

    eq2 = read(file_name)
    os.remove(file_name)

    assert np.isclose(equilibrium.magnetic_axis.R, eq2.magnetic_axis.R, atol=1e-5, rtol=1e-4)
    assert np.isclose(equilibrium.magnetic_axis.Z, eq2.magnetic_axis.Z, atol=1e-5, rtol=1e-4)


def test_fiesta_gfile_vs_database():
    tmp_dir = tempfile.TemporaryDirectory()
    file_name = '{}/g0000.0000'.format(tmp_dir.name)

    resource_package = "pleque.resources"
    gfile_file = pkg_resources.resource_filename(resource_package, "test00.gfile")
    nc_file = pkg_resources.resource_filename(resource_package, "test00.nc")

    basedata = xr.load_dataset(nc_file).load()

    eq_nc = pleque.Equilibrium(basedata, cocos=13)
    eq_gfile = read_geqdsk(gfile_file)

    # with open(gfile_file, "r") as f:
    #     eq_gfile = _geqdsk.read(f)

    eq_nc.to_geqdsk(file_name, q_positive=True)
    eq2 = read_geqdsk(file_name)

    def compare_two(eq_1: pleque.Equilibrium, eq_2: pleque.Equilibrium):
        assert np.isclose(np.abs(eq_1.I_plasma), np.abs(eq_2.I_plasma), atol=1e4, rtol=1e-3)
        assert np.isclose(eq_1.pressure(psi_n=0.2)[0], eq_2.pressure(psi_n=0.2)[0], atol=3e3, rtol=1e-2)
        assert np.isclose(np.abs(eq_1.BvacR), np.abs(eq_2.BvacR))
        assert np.isclose(np.abs(eq_1.q(psi_n=0.5)[0]), np.abs(eq_2.q(psi_n=0.5)[0]), atol=1e-3, rtol=1e-3)

    compare_two(eq_nc, eq_gfile)
    compare_two(eq_gfile, eq2)
    compare_two(eq_nc, eq2)
