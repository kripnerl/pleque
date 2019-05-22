import h5py
import numpy as np
import pkg_resources
import xarray as xr

from pleque.core import Equilibrium
from pleque.io._geqdsk import read, data_as_ds
from pleque.io.tools import EquilibriaTimeSlices


def cdb(shot=None, time=1060, revision=1):
    """


    :param shot: number of shot in cdb, defaults to last
    :param time: closest time [ms] of target equilibrium, defaults to 10 ms after shaping
                 if None then an EFITSlices instance is returned
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


def read_efithdf5(file_path, time=None):
    """
    Loads Equilibrium information from an efit file.

    :param file_path: path to the hdf5 compass efit file
    :param time: closest time [ms] of target equilibrium,
                 if None then an EFITSlices instance is returned
    :return: Equilibrium
    """

    with h5py.File(file_path, 'r') as f5efit:  # open EFITXX.rev.h5

        t = f5efit['time'][:]
        if t[0] < 100:  # heuristic, first time should be above 100 if in ms
            t *= 1e3  # put into ms
        dst = xr.Dataset({
            'psi': (['time', 'R', 'Z'], f5efit['output/profiles2D/poloidalFlux']),
            'pressure': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/staticPressure']),
            'pprime': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/staticPPrime']),
            'F': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/rBphi']),
            'FFprime': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/ffPrime']),
            'q': (['time', 'psi_n'], f5efit['output/fluxFunctionProfiles/q']),

        }, coords={
            'time': t,
            'Rt': (['time', 'R'], f5efit['output/profiles2D/r']),
            'Zt': (['time', 'Z'], f5efit['output/profiles2D/z']),
            'psi_n': (['psi_n'], f5efit['output/fluxFunctionProfiles/normalizedPoloidalFlux']),
        }
        )
        # the limiter is not expected to change in time, so take 0th time index
        limiter = np.column_stack([f5efit['input/limiter/{}Values'.format(x)][0, :]
                                   for x in 'rz'])
        dst.load()

    efit_slices = EquilibriaTimeSlices(dst, limiter)
    if time is not None:
        eq = efit_slices.get_time_slice(time)
        return eq
    else:
        return efit_slices


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

    eq._Ip = ds.attrs['cpasma']

    return eq
