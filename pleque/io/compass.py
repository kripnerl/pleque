import numpy as np

from pleque.core import Equilibrium

import h5py
import xarray as xr
from pleque.io._geqdsk import read, data_as_ds
import pkg_resources

def cdb(shot=None, time=1060, revision=1):
    """


    :param shot: number of shot in cdb, defaults to last
    :param time: closest time [ms] of target equilibrium, defaults to 10 ms after shaping
    :param revision: EFIT revision, defaults to first (post-shot standard)
    :return: Equilibrium
    """

    import pyCDB.client

    cdb = pyCDB.client.CDBClient()
    if shot is None:
        shot = cdb.last_shot_number()
    # psi_RZ generic ID
    sig_ref = cdb.get_signal_references(record_number=shot,
                                        generic_signal_id=2860,
                                        revision=revision)[0]
    data_ref = cdb.get_data_file_reference(**sig_ref)
    eq = read_efithdf5(data_ref.full_path, time=time)

    return eq


def read_efithdf5(file_path, time):
    """
    Loads Equilibrium information from an efit file.

    :param file_path: path to the hdf5 compass efit file
    :param time: closest time [ms] of target equilibrium, defaults to 10 ms after shaping
    :return: Equilibrium
    """

    with h5py.File(file_path, 'r') as f5efit:  # open EFITXX.rev.h5

        t = f5efit['time'].value
        if t[0] < 100:  # heuristic, first time should be above 100 if in ms
            t *= 1e3  # put into ms
        dst = xr.Dataset({
            'psi': (['time', 'R', 'Z'], f5efit['output/profiles2D/poloidalFlux']),
            'pressure': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/staticPressure']),
            'pprime': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/staticPPrime']),
            'F': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/rBphi']),
            'ffprime': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/ffPrime']),
            'qpsi': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/q']),

        }, coords={
            'time': t,
            'Rt': (['time', 'R'], f5efit['output/profiles2D/r']),
            'Zt': (['time', 'Z'], f5efit['output/profiles2D/z']),
            'psi_n': f5efit['output/fluxFunctionProfiles/normalizedPoloidalFlux'],
        }
        )
        ds = dst.sel(time=time, method='nearest').rename({'Rt': 'R', 'Zt': 'Z'})
        # limiter is not expected to change in tome, so take 0th time index
        limiter = np.column_stack([f5efit['input/limiter/{}Values'.format(x)][0, :]
                                   for x in 'rz'])
        eq = Equilibrium(ds, limiter)
    return eq


def read_fiesta_equilibrium(filepath, first_wall=None):
    """
    Current versions of the equilibria are stored in
    `/compass/Shared/Common/COMPASS-UPGRADE/RP1 Design/Equilibria/v3.1`

    :param filepath: Path to fiesta g-file equilibria
    :param first_wall: Path to datafile with limiter line. (Fiesta doesn't store limiter contour into g-file).
        If `None` IBA limiter v 3.1 is taken.
    :return: Equilibrium: Instance of `Equilibrium`
    """

    resource_package = 'pleque'

    # ds = readeqdsk_xarray(filepath)
    with open(filepath, 'r') as f:
        data = read(f)
        ds = data_as_ds(data)

    # If there are some limiter data. Use them as and limiter.
    if 'r_lim' in ds and 'z_lim' in ds and ds.r_lim.size > 3 and first_wall is None:
        first_wall = np.stack((ds.r_lim.data, ds.z_lim.data)).T

    if first_wall is None:
        print('--- No limiter specified. The IBA v3.1 limiter will be used.')
        first_wall = 'resources/limiter_v3_1_iba_v2.dat'
        first_wall = pkg_resources.resource_filename(resource_package, first_wall)

    if isinstance(first_wall, str):
        first_wall = np.loadtxt(first_wall)

    eq = Equilibrium(ds, first_wall=first_wall)

    # todo: now assume cocos = 3 => q < 0
    if np.sum(ds.qpsi.data) > 0:
        qpsi = ds.qpsi.data * -1
    else:
        qpsi = ds.qpsi.data

    # eq._q_spl = UnivariateSpline(ds.psi_n.data, ds.qpsi.data, s=0, k=3)
    # eq._q_spl = UnivariateSpline(ds.psi_n.data, qpsi, s=0, k=3)
    # eq._dq_dpsin_spl = eq._q_spl.derivative()
    # eq._q_anideriv_spl = eq._q_spl.antiderivative()
    eq.I_plasma = ds.attrs['cpasma']

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

    # Equilibrium.q = q
    # Equilibrium.diff_q = diff_q
    # Equilibrium.tor_flux = tor_flux

    return eq
