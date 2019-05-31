import pytest
pycdb = pytest.importorskip('pyCDB')

def test_cdb():
    from pleque.io.compass import read_efithdf5
    from pleque.io.compass import cdb
    from os.path import expanduser


    eq = cdb(17636, 1125)
    eq = cdb(17854, 1000)
    #eq = read_efithdf5(expanduser("~/EFIT/17636.1.h5"), time=1125)

    print(eq)


def test_cdb_EFITSlices():
    from pleque.io.compass import cdb
    from pleque.io.tools import EquilibriaTimeSlices

    efit_slices = cdb(17636, time=None)
    assert isinstance(efit_slices, EquilibriaTimeSlices)
    eq = efit_slices.get_time_slice(1125)
    print(eq)
