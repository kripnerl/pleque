import numpy as np
import xarray as xr

from jet.data import sal

from sal.core.exception import NodeNotFound

from pleque import Equilibrium


def deltapsi_calc(pulse):
    """
    Calculates difference between value of psi at magnetic axis and at sepraratrix, function of time

    :param pulse: JET pulse number
    :return : difference, function of time
    """
    DATA_PATH = '/pulse/{}/ppf/signal/jetppf/efit/{}:{}'
    psi_lcfs = sal.get(DATA_PATH.format(pulse, 'fbnd', 0))
    psi_axis = sal.get(DATA_PATH.format(pulse, 'faxs', 0))
    deltapsi = (psi_lcfs.data - psi_axis.data)
    return deltapsi


def pprime_calc(pressure, deltapsi, lenpsin):
    """
    Calculates derivative of pressure along psi coordinate
    :param pressure: pressure signal from database
    :param deltapsi: result of deltapsi_calc
    :param lenpsin: 1/number of points in psi_n axis
    :return: the dp/dpsi on axis time and psi_n
    """
    pprime = deltapsi[:, np.newaxis]*np.gradient(pressure.data, 1/lenpsin, axis=1)
    return pprime


def ffprime_calc(f, deltapsi, lenpsin):
    """
    Calculates ffprime
    :param f: f function
    :param deltapsi: result of deltapsi_calc
    :param lenpsin: 1/number of points in psi_n axis
    :return: the f*df/fpsi on axis time and psi_n
    """
    ffprime = deltapsi[:,np.newaxis]*f.data*np.gradient(f.data, 1/lenpsin,axis=1)
    return ffprime

def sal_jet(pulse, timex=47.0):
    """
    Main loading routine, based on simple access layer, loads ppf data, calculates derivatives
    :param pulse: JET pulse number
    :param timex: time of slice
    :return: equilibrium
    """

    data_path = '/pulse/{}/ppf/signal/jetppf/efit/{}:{}'

    # default sequence
    sequence = 0

    # obtain psi data (reshape, transpose) and time axis
    packed_psi = sal.get(data_path.format(pulse, 'psi', sequence))
    psi = packed_psi
    psi.data = packed_psi.data[:, :].reshape(len(packed_psi.dimensions[0]), 33, 33)
    psi.data = np.swapaxes(psi.data, 1, 2)

    time = packed_psi.dimensions[0].data

    # psi grid axis
    r = sal.get(data_path.format(pulse, 'psir', sequence)).data
    z = sal.get(data_path.format(pulse, 'psiz', sequence)).data

    # pressure profile
    pressure = sal.get(data_path.format(pulse, 'p', sequence))
    psi_n = pressure.dimensions[1].data

    # f-profile
    f = sal.get(data_path.format(pulse, 'f', sequence))

    # q-profile
    qpsi = sal.get(data_path.format(pulse, 'q', sequence))

    # calculate pprime and ffprime
    deltapsi = deltapsi_calc(pulse)

    pprime = pprime_calc(pressure, deltapsi, len(psi_n))

    ffprime = ffprime_calc(f, deltapsi, len(psi_n))

    #create dataset

    dst = xr.Dataset({
        'psi': (['time', 'R', 'Z'], psi.data),
        'pressure': (['time', 'psi_n'], pressure.data),
        'pprime': (['time', 'psi_n'], pprime),
        'fpol': (['time', 'psi_n'], f.data),
        'ffprime': (['time', 'psi_n'], ffprime),
        'qpsi': (['time', 'psi_n'], qpsi.data),
        'R': (['R'], r),
        'Z': (['Z'], z),

    }, coords={
        'time': time,
        'psi_n': psi_n,
    }
    )

    # select desired time
    ds = dst.sel(time=timex, method='nearest')

    # try to load limiter from ppfs

    try:
        limiter_r = sal.get(data_path.format(pulse, 'rlim', sequence)).data.T
        limiter_z = sal.get(data_path.format(pulse, 'zlim', sequence)).data.T
    except NodeNotFound:
        limiter_r = None
        limiter_z = None

    limiter = np.column_stack([limiter_r, limiter_z])

    # create pleque equilibrium

    eq = Equilibrium(ds, limiter)

    return eq
