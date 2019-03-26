import numpy as np
import xarray as xr
import h5py

from pleque.core import Coordinates, FluxFunction


def read(equilibrium, file, time):
    """
    Reads Metis simulation and returns FluxFunc object containing 1D profiles
    :param equilibrium: Pleque equilibrium object to connect to the metis profiles
    :param file: Metis output file saved as matlab hdf5 file
    :param time: Discharge time specifying which profiles will be loaded. Currently only a single time is supported.
    :return: FluxFuncs instance of `equilibrium`
    """
    # TODO: Shall whe save 0D values from metis into the FluxFunc object?


    fluxfun = equilibrium.fluxfuncs

    # 1D profiles
    with h5py.File(file, "r") as f:

        times = np.array(f["post/zerod/temps"]).squeeze()

        ds = xr.Dataset()

        prof = f["post/profil0d"]

        xli = np.array(prof["xli"]).squeeze()

        for name in prof.keys():
            if prof[name].size == times.size * xli.size:
                da = xr.DataArray(np.array(prof[name]).squeeze(), coords=[xli, times], dims=["x", "time"])
                ds[name] = da

    toexp = ds.sel(time=time, method="nearest").drop("time")
    psi_axis = toexp.psi.values[np.argmin(np.abs(toexp.psi.values))]
    psi_sep = toexp.psi.values[np.argmax(np.abs(toexp.psi.values))]
    psi_n = np.abs((toexp.psi.values - psi_axis) / (psi_sep - psi_axis))

    crds = Coordinates(equilibrium=equilibrium, psi_n=psi_n)

    for name in list(toexp.variables.keys()):
        #todo: better handling of complex values
        if np.all(np.isreal(toexp[name].values)):
            fluxfun.add_flux_func(name, toexp[name].values, crds)

    return fluxfun


def to_omas(fluxfunc, coordinates, ods=None, time=np.array([0], ndmin=1)):
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

    ods["core_profiles"]["ids_properties"]["homogeneous_time"] = 1  # this has to be there for ["core_profiles"]["time"]
    ods["core_profiles"]["time"] = np.array(time, ndmin=1)

    # rho toroidal profiles
    ods["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor"] = fluxfunc._equi.tor_flux(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"] = (fluxfunc._equi.tor_flux(coordinates) /
                                                                      fluxfunc._equi.tor_flux(psi_n=1))

    # plasma profiles
    ods["core_profiles"]["profiles_1d"][0]["ion"][0]["density"] = fluxfunc.nip(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["ion"][0]["temperature"] = fluxfunc.tip(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["ion"][0]["velocity"]["toroidal"] = fluxfunc.vtor(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["electrons"]["density"] = fluxfunc.nep(coordinates)
    ods["core_profiles"]["profiles_1d"][0]["electrons"]["temperature"] = fluxfunc.tep(coordinates)

    return ods
