import numpy as np

from pleque import Equilibrium


def read_fiesta_equilibrium(filepath, first_wall=None):
    """
    Current versions of the equilibria are stored in
    `/compass/Shared/Common/COMPASS-UPGRADE/RP1 Design/Equilibria/v3.1`

    :param filepath: Path to fiesta g-file equilibria
    :param first_wall: Path to datafile with limiter line. (Fiesta doesn't store limiter contour into g-file). If `None` IBA limiter v 3.1 is taken.
    :return: Equilibrium: Instance of `Equilibrium`
    """
    from pleque.io.readgeqdsk import readeqdsk_xarray
    from scipy.interpolate import UnivariateSpline
    import pkg_resources

    resource_package = __name__


    if first_wall is None:
        first_wall = '../../test/test_files/compu/limiter_v3_1_iba.dat'
        first_wall = pkg_resources.resource_filename(resource_package, first_wall)

    ds = readeqdsk_xarray(filepath)
    first_wall = np.loadtxt(first_wall)

    eq = Equilibrium(ds, first_wall=first_wall)

    eq._q_spl = UnivariateSpline(ds.psi_n.data, ds.qpsi.data, s=0, k=3)
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