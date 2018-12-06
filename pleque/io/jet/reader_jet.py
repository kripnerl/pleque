import numpy as np
import xarray as xr

from jet.data import sal

from sal.core.exception import NodeNotFound

from pleque import Equilibrium


def psin2psi(pulse, psi_n):
    DATA_PATH = '/pulse/{}/ppf/signal/jetppf/efit/{}:{}'
    psi_lcfs = sal.get(DATA_PATH.format(pulse, 'fbnd', 0))
    psi_axis = sal.get(DATA_PATH.format(pulse, 'faxs', 0))
    scaling = (psi_lcfs.data - psi_axis.data)
    psi_1D = np.outer(psi_n, np.ones(len(psi_axis.dimensions[0].data))).T * scaling[:, np.newaxis] + psi_axis.data[:,
                                                                                                     np.newaxis]

    # print(np.shape(np.outer(psi_n,np.ones(len(psi_axis.dimensions[0].data))).T))
    print(np.shape(psi_1D))
    return psi_1D


def pprime_ugly(pressure, psi1D):
    print(np.shape(pressure), psi1D[10, :])
    pprime = [np.gradient(pressure.data[i, :], psi1D[i, :]) for i in range(0, np.shape(pressure.data)[0])]
    return np.vstack(pprime)


def ffprime_ugly(f, psi1D):
    print(np.shape(f), psi1D[10, :])
    ffprime = [f.data[i, :] * np.gradient(f.data[i, :], psi1D[i, :]) for i in range(0, np.shape(f.data)[0])]
    return np.vstack(ffprime)


def sal_jet(pulse, timex=47.0):
    DDA_PATH = '/pulse/{}/ppf/signal/jetppf/efit:{}'
    DATA_PATH = '/pulse/{}/ppf/signal/jetppf/efit/{}:{}'

    # defaults
    sequence = 0

    # obtain psi data and timebase
    packed_psi = sal.get(DATA_PATH.format(pulse, 'psi', sequence))
    psi = packed_psi
    psi.data = packed_psi.data[:, :].reshape(len(packed_psi.dimensions[0]), 33, 33)
    psi.data = np.swapaxes(psi.data, 1, 2)

    time = packed_psi.dimensions[0].data
    # get time index of selected time
    time_ind = np.argmin(psi.dimensions[0].data - timex)

    # psi grid axis
    r = sal.get(DATA_PATH.format(pulse, 'psir', sequence)).data
    z = sal.get(DATA_PATH.format(pulse, 'psiz', sequence)).data

    # pressure profile
    pressure = sal.get(DATA_PATH.format(pulse, 'p', sequence))
    psi_n = pressure.dimensions[1].data

    # f-profile
    f = sal.get(DATA_PATH.format(pulse, 'f', sequence))

    # qprofile
    qpsi = sal.get(DATA_PATH.format(pulse, 'q', sequence))


    psi1D = psin2psi(pulse, psi_n)

    pprime = pprime_ugly(pressure, psi1D)


    ffprime = ffprime_ugly(f, psi1D)


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
    ds = dst.sel(time=timex, method='nearest')
    # limiter is not expected to change in tome, so take 0th time index

    try:
        limiter_r = sal.get(DATA_PATH.format(pulse, 'rlim', sequence)).data.T
        limiter_z = sal.get(DATA_PATH.format(pulse, 'zlim', sequence)).data.T
    except NodeNotFound:
        limiter_r = None
        limiter_z = None

    limiter = np.column_stack([limiter_r, limiter_z])

    eq = Equilibrium(ds, limiter)

    return eq
