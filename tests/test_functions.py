from pleque.io import _geqdsk
import numpy as np
import xarray as xa

from pleque.utils.equi_tools import pprime2p, ffprime2f


def test_profiles_integration(geqdsk_file):
    with open(geqdsk_file, 'r') as f:
        eq_in = _geqdsk.read(f)

    psi_ax = eq_in['simagx']
    psi_bnd = eq_in['sibdry']

    p = eq_in['pres']

    if np.isclose(p[-1], 0):
        psi_n = np.linspace(0, 1, len(p), endpoint=True)
    else:
        psi_n = np.linspace(0, 1, len(p), endpoint=False)

    pprime = eq_in['pprime']
    pprime = xa.DataArray(pprime, [psi_n], ['psi_n'])

        ffprime = eq_in['FFprime']
        f = eq_in['F']

        f0 = f[-1]

        p_calc = pprime2p(pprime, psi_ax, psi_bnd)
        f_calc = ffprime2f(ffprime, psi_ax, psi_bnd, f0)

        assert np.allclose(p, p_calc)
        assert np.allclose(f, f_calc)

