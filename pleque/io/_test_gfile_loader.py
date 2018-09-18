def read_gfile(g_file: str, limiter: str = None):
    from tokamak.formats import geqdsk
    import numpy as np
    import xarray as xa
    from ..core import Equilibrium
    from scipy.interpolate import UnivariateSpline

    eq_gfile = geqdsk.read(g_file)

    psi = eq_gfile['psi']
    r = eq_gfile['r'][:, 0]
    z = eq_gfile['z'][0, :]

    pressure = eq_gfile['pressure']
    fpol = eq_gfile['fpol']
    qpsi = eq_gfile['qpsi']

    psi_n = np.linspace(0, 1, len(fpol))

    eq_ds = xa.Dataset({'psi': (['R', 'Z'], psi),
                        'pressure': ('psi_n', pressure),
                        'fpol': ('psi_n', fpol)},
                       coords={'R': r,
                               'Z': z,
                               'psi_n': psi_n})
    if limiter is not None:
        lim = np.loadtxt(limiter)
        print(lim.shape)
    else:
        lim = None

    eq = Equilibrium(eq_ds, first_wall=lim)

    eq._geqdsk = eq_gfile

    eq._q_spl = UnivariateSpline(psi_n, qpsi, s=0, k=3)
    eq._dq_dpsin_spl = eq._q_spl.derivative()
    eq._q_anideriv_spl = eq._q_spl.antiderivative()

    def q(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        if R is not None and Z is not None:
            psi_n = self.psi_n(R=R, Z=Z, grid=grid)
        return self._q_spl(psi_n)

    def diff_q(self: eq, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        """

        :param self:
        :param coordinates:
        :param R:
        :param Z:
        :param psi_n:
        :param coord_type:
        :param grid:
        :param coords:
        :return: Derivative of q with respect to psi.
        """
        if R is not None and Z is not None:
            psi_n = self.psi_n(R=R, Z=Z, grid=grid)
        return self._dq_dpsin_spl(psi_n) * self._diff_psiN

    def tor_flux(self: eq, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        if R is not None and Z is not None:
            psi_n = self.psi_n(R=R, Z=Z, grid=grid)

        return eq._q_anideriv_spl(psi_n) * (1 / self._diff_psi_n)

    # eq.q = q
    # eq.diff_q = diff_q
    # eq.tor_flux = tor_flux

    Equilibrium.q = q
    Equilibrium.diff_q = diff_q
    Equilibrium.tor_flux = tor_flux

    return eq
