"""
In case of any issues, questions and comments contact

Ondrej Ficker

ficker@ipp.cas.cz

"""

import numpy as np
import xarray as xr

from jet.data import sal

from sal.core.exception import NodeNotFound

from pleque.core import Equilibrium


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


def FFprime_calc(F, deltapsi, lenpsin):
    """
    Calculates FFprime
    :param F: f function
    :param deltapsi: result of deltapsi_calc
    :param lenpsin: 1/number of points in psi_n axis
    :return: the f*df/fpsi on axis time and psi_n
    """
    FFprime = deltapsi[:,np.newaxis] * F.data * np.gradient(F.data, 1 / lenpsin, axis=1)
    return FFprime

def sal_jet(pulse, timex=47.0, time_unit="s"):
    """
    Main loading routine, based on simple access layer, loads ppf data, calculates derivatives
    :param pulse: JET pulse number
    :param timex: time of slice
    :param time_unit: (str) "s" or "ms"
    :return: equilibrium
    """

    if time_unit.lower() == "s":
        time_factor = 1.
    elif time_unit.lower() == "ms":
        time_factor = 1000.
    else:
        raise ValueError("Unknown `time_unit`.")


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
    q = sal.get(data_path.format(pulse, 'q', sequence))

    # calculate pprime and FFprime
    deltapsi = deltapsi_calc(pulse)

    pprime = pprime_calc(pressure, deltapsi, len(psi_n))

    FFprime = FFprime_calc(f, deltapsi, len(psi_n))

    #create dataset

    dst = xr.Dataset({
        'psi': (['time', 'R', 'Z'], psi.data),
        'pressure': (['time', 'psi_n'], pressure.data),
        'pprime': (['time', 'psi_n'], pprime),
        'F': (['time', 'psi_n'], f.data),
        'FFprime': (['time', 'psi_n'], FFprime),
        'q': (['time', 'psi_n'], q.data),
        'R': (['R'], r),
        'Z': (['Z'], z),

    }, coords={
        'time': time,
        'psi_n': psi_n,
    }
    )

    # select desired time
    ds = dst.sel(time=timex/time_factor, method='nearest')

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
