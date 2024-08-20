import pytest

pycdb = pytest.importorskip('pyCDB')

import numpy as np
import os
import tempfile

os.environ['CDB_PATH'] = os.getenv('CDB_PATH', '/home/kripner/Projects/CDB/src')


@pytest.mark.parametrize("shot,variant,revision,time",
                         [(17636, '', -1, 1125),
                          (17854, '', -1, 1000),
                          (11399, 'v4_std_O', 1, 1060),
                          (21267, 'V4_std_O', -1, 1400),
                          (17588, 't1100_ZsepTS_285', 2, 1100)
                          ])
def test_cdb(shot, variant, revision, time):
    from pleque.io.compass import read_efithdf5
    from pleque.io.compass import cdb
    from os.path import expanduser

    eq = cdb(shot=shot, time=time, variant=variant, revision=revision)

    assert eq is not None
    # eq = read_efithdf5(expanduser("~/EFIT/17636.1.h5"), time=1125)

    print(eq)


def test_cdb_EFITSlices():
    from pleque.io.compass import cdb
    from pleque.io.tools import EquilibriaTimeSlices

    efit_slices = cdb(17636, time=None)
    assert isinstance(efit_slices, EquilibriaTimeSlices)
    eq = efit_slices.get_time_slice(1125)
    print(eq)


def test_cdb_to_gfile():
    from pleque.io.compass import cdb
    from pleque.io import readers

    eq = cdb(17854, 1000)

    tmp_dir = tempfile.gettempdir()

    file = tmp_dir + '/cbd_eqdsk'
    eq.to_geqdsk(file)
    eq_gfile = readers.read_geqdsk(file)

    assert np.isclose(eq.I_plasma, eq_gfile.I_plasma, atol=1e6, rtol=1e-2)
    assert np.isclose(eq.B_tor(eq.magnetic_axis), eq_gfile.B_tor(eq_gfile.magnetic_axis), rtol=1e-4)

    file = tmp_dir + '/cbd_eqdsk2'
    eq.to_geqdsk(file, use_basedata=True)
    eq_gfile2 = readers.read_geqdsk(file)

    assert np.isclose(eq.I_plasma, eq_gfile2.I_plasma, atol=1e6, rtol=1e-2)
    assert np.isclose(eq.B_tor(eq.magnetic_axis), eq_gfile2.B_tor(eq_gfile2.magnetic_axis), rtol=1e-2)

    assert np.allclose(eq_gfile.magnetic_axis.as_array(), eq_gfile2.magnetic_axis.as_array(), rtol=1e-2)


@pytest.mark.parametrize("shot,variant,revision", [(3100, '', -1),
                                                   (6400, '', -1),
                                                   (24300, '', -1),
                                                   (24300, 'nice_currents', -1),
                                                   pytest.param(24300, '', -2,
                                                                marks=pytest.mark.xfail(reason="Bug in CDB")),
                                                   (7400, '', -1)])
def test_cudb(shot, variant, revision, time=1.0):
    from pleque.io.compass import cudb

    eq = cudb(shot=shot, time=time, variant=variant, revision=revision)

    # Check if the equilibrium is valid
    assert 4.0e6 > np.abs(eq.I_plasma) > 0.1e6, f"Invalid plasma current: {eq.I_plasma} kA"


@pytest.mark.parametrize("use_basedata", [True, False])
def test_cudb_to_gfile(use_basedata):
    from pleque.io import compass, readers

    eq = compass.cudb(6400, 2.0)

    tmp_dir = tempfile.gettempdir()

    file = tmp_dir + '/cubd_eqdsk'
    eq.to_geqdsk(file, use_basedata=use_basedata)
    eq_gfile = readers.read_geqdsk(file)

    assert np.isclose(eq.I_plasma, eq_gfile.I_plasma, atol=1e6, rtol=1e-2)
    assert np.isclose(eq.B_tor(eq.magnetic_axis), eq_gfile.B_tor(eq_gfile.magnetic_axis), rtol=1e-4)
