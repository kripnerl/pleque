import numpy as np

from pleque import Equilibrium


def read_fiesta_equilibrium(filepath, first_wall=None):
    """
    Current versions of the equilibria are stored in
    `/compass/Shared/Common/COMPASS-UPGRADE/RP1 Design/Equilibria/v3.1`

    :param filepath: Path to fiesta g-file equilibria
    :param first_wall: Path to datafile with limiter line. (Fiesta doesn't store limiter contour into g-file).
        If `None` IBA limiter v 3.1 is taken.
    :return: Equilibrium: Instance of `Equilibrium`
    """
    from pleque.io._readgeqdsk import readeqdsk_xarray
    from scipy.interpolate import UnivariateSpline
    import pkg_resources

    resource_package = __name__

    ds = readeqdsk_xarray(filepath)

    # If there are some limiter data. Use them as and limiter.
    if 'r_lim' in ds and 'z_lim' in ds and ds.r_lim.size > 3 and first_wall is None:
        first_wall = np.stack(ds.r_lim.data, ds.z_lim.data)

    if first_wall is None:
        print('--- No limiter specified. The IBA v3.1 limiter will be used.')
        first_wall = '../../test/test_files/compu/limiter_v3_1_iba.dat'
        first_wall = pkg_resources.resource_filename(resource_package, first_wall)

    if isinstance(first_wall, str):
        first_wall = np.loadtxt(first_wall)

    eq = Equilibrium(ds, first_wall=first_wall)

    #todo: now assume cocos = 3 => q < 0
    if np.sum(ds.qpsi.data) > 0:
        qpsi = ds.qpsi.data * -1
    else:
        qpsi = ds.qpsi.data


    #eq._q_spl = UnivariateSpline(ds.psi_n.data, ds.qpsi.data, s=0, k=3)
    eq._q_spl = UnivariateSpline(ds.psi_n.data, qpsi, s=0, k=3)
    eq._dq_dpsin_spl = eq._q_spl.derivative()
    eq._q_anideriv_spl = eq._q_spl.antiderivative()

    # noinspection PyPep8Naming
    def q(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._q_spl(coord.psi_n)

    # noinspection PyPep8Naming
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
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._dq_dpsin_spl(coord.psi_n) * self._diff_psiN

    # noinspection PyPep8Naming
    def tor_flux(self: eq, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return eq._q_anideriv_spl(coord.psi_n) * (1 / self._diff_psi_n)

    # eq.q = q
    # eq.diff_q = diff_q
    # eq.tor_flux = tor_flux

    Equilibrium.q = q
    Equilibrium.diff_q = diff_q
    Equilibrium.tor_flux = tor_flux

    return eq
