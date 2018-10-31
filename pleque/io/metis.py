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
    #fluxfun = FluxFuncs(equi=equilibrium)
    fluxfun = equilibrium.fluxfuncs

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

def to_omas(fluxfunc, coordinates, ods = None, time = np.array([0], ndmin=1)):
    """

    :param fluxfunc: FluxFunction with metis profiles
    :param coordinates: 1D coordinates specifying the independent variable (allows control over resolution).
    :param ods: omas ODS file
    :param time: ods data strucutr time to be used
    :return: omas ods data structure
    """
    # TODO: So far only profiles to be used by ASCOT are saved. Others will be added later (or upon request)
    import omas

    if ods is None:
        ods = omas.ODS()

    ods["core_profiles"]["ids_properties"]["homogeneous_time"] = 1  #this has to be there for ["core_profiles"]["time"]
    ods["core_profiles"]["time"] = np.array(time, ndmin=1)

    #rho toroidal profiles
    ods["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor"] = fluxfunc._equi.tor_flux(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"] = (fluxfunc._equi.tor_flux(coordinates) /
                                                                      fluxfunc._equi.tor_flux(psi_n=1))

    #plasma profiles
    ods["core_profiles"]["profiles_1d"][0]["ion"][0]["density"] = fluxfunc.nip(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["ion"][0]["temperature"] = fluxfunc.tip(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["ion"][0]["velocity"]["toroidal"] = fluxfunc.vtor(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["electrons"]["density"] = fluxfunc.nep(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["electrons"]["temperature"] = fluxfunc.tep(coordinates)

    return ods