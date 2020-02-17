import pytest
pycdb = pytest.importorskip('pyCDB')

import numpy as np
import os

os.environ['CDB_PATH'] = os.getenv('CDB_PATH', '/home/kripner/Projects/CDB/src')

def test_cdb():
    from pleque.io.compass import read_efithdf5
    from pleque.io.compass import cdb
    from os.path import expanduser


    eq = cdb(17636, 1125)
    eq = cdb(17854, 1000)
    eq = cdb(11399, variant='v4_std_O', revision=1)
    #eq = read_efithdf5(expanduser("~/EFIT/17636.1.h5"), time=1125)

    print(eq)


def test_cdb_EFITSlices():
    from pleque.io.compass import cdb
    from pleque.io.tools import EquilibriaTimeSlices

    efit_slices = cdb(17636, time=None)
    assert isinstance(efit_slices, EquilibriaTimeSlices)
    eq = efit_slices.get_time_slice(1125)
    print(eq)


def test_cudb():
    from pleque.io.compass import cudb

    eq = cudb(6400, 2.0)

    assert np.isclose(np.abs(eq.I_plasma), 2e6, atol=1e-2, rtol=1e-2)

    # shots = [3100, 3109, 3400, 4400, 6400, 6408, 6409, 6600, 7400]
    # shot_time = 2.0
    # currents = []
    #
    # for shot in shots:
    #     eq = cudb(shot, shot_time)
    #     currents.append(eq.I_plasma)
    #
    # currents = np.abs(currents)
    #
    # assert np.all(currents < 3.0e6)
    # assert np.all(currents > 0.2e6)


def test_cudb_to_gfile():
    from pleque.io import compass, readers
    import tempfile

    eq = eq = compass.cudb(6400, 2.0)

    tmp_dir = tempfile.gettempdir()
    print(tmp_dir)

    file = tmp_dir + '/cubd_eqdsk'

    eq.to_geqdsk(file)

    eq_gfile = readers.read_geqdsk(file)

    assert np.isclose(eq.I_plasma, eq_gfile.I_plasma)
    assert np.isclose(eq.B_tor(eq.magnetic_axis), eq_gfile.B_tor(eq_gfile.magnetic_axis))
