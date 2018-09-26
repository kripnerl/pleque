import matplotlib.pyplot as plt
from pleque.core import FluxFuncs, Coordinates

from test.testing_utils import load_testing_equilibrium
import pydons
import os
import xarray as xr
import numpy as np


def read(equilibrium,file, time):
    """
    Reads Metis simulation and returns FluxFunc object containing 1D profiles
    :param equilibrium: Pleque equilibrium object to connect to the metis profiles
    :param file: Metis output file saved as matlab hdf5 file
    :param time: Discharge time specifying which profiles will be loaded. Currently only a single time is supported.
    :return: FluxFunc object
    """
    #TODO: Shall whe save 0D values from metis into the FluxFunc object?

    post = pydons.loadmat(file)["post"]
    fluxfun = FluxFuncs(equi=equilibrium)

    zerod = post.zerod
    times = zerod.temps.squeeze()
    ds = xr.Dataset()

    #0D values
    #for name in zerod.keys():
    #    if zerod[name].size == times.size:
    #        da = xr.DataArray(zerod[name].squeeze(), coords=[times], dims=["time"])
    #        ds[name] = da


    # 1D profiles
    prof = post.profil0d
    xli = prof.xli.squeeze()
    for name in prof.keys():
        if prof[name].size == times.size * xli.size:
            da = xr.DataArray(prof[name].squeeze(), coords=[times, xli], dims=["time", "x"])
            ds[name] = da

    toexp =ds.sel(time=time, method="nearest").drop("time")
    psi_axis = toexp.psi.values[np.argmin(np.abs(toexp.psi.values))]
    psi_sep = toexp.psi.values[np.argmax(np.abs(toexp.psi.values))]
    psi_n = np.abs((toexp.psi.values - psi_axis)/(psi_sep - psi_axis))

    crds = Coordinates(equilibrium=equilibrium,psi_n=psi_n)

    for name in  list(toexp.variables.keys()):
        print(name)
        fluxfun.add_flux_func(name, toexp[name].values,crds)

    return fluxfun