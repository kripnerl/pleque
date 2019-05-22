import xarray as xr
from collections import OrderedDict
from typing import Union
from pleque import Equilibrium


class EquilibriaTimeSlices:
    """
    Simple container for a series of time-slices from xr Dataset with equilibria's data.
    *Note:* This is temporary solution before implementation of support of time-evolving equilibrium.
    """

    def __init__(self, eqs_dataset, limiter):
        """Create instance for generating equilibria at given times

        :param eqs_dataset: Dataset containing time-dependent equilibira inputs
        "param limiter: time-independent limiter [R,Z] coords
        """
        self.eqs_dataset = eqs_dataset
        self.limiter = limiter

    def get_time_slice(self, time: float):
        """Creates an Equilibrium from the slice nearest to the specified time"""
        ds = self.eqs_dataset.sel(time=time, method='nearest').rename(
            {'Rt': 'R', 'Zt': 'Z'})
        eq = Equilibrium(ds, self.limiter)
        return eq


def xr2dict(ds: Union[xr.Dataset, xr.DataArray]):
    """
    Convert Dataset or DataArray to single dictionary.

    :param ds: [xr.Dataset, xr.DataArray]
    :return: (dict) Dataset or DataArray converted into single OrderedDict.
    """

    ret_dict = OrderedDict()
    ret_dict.update(ds.variables)
    ret_dict.update(ds.attrs)

    return ret_dict


def da2dict(da: xr.DataArray):
    """
    Return representation of xarray DataArray as dictionary.
    If DataArray has no name, data are returned under key "data".

    :param da: DataArray
    :return: dictionary with data from `da`
    """

    ret_dict = dict()
    ret_dict.update(da.attrs)

    # add data:
    if da.name is None:
        ret_dict["data"] = da.data
    else:
        ret_dict[da.name] = da.data

    # Add axes:
    for k, val in da.coords.items():
        ret_dict[k] = val.data
        # axis attributes:

        for ka, atr in val.attrs.items():
            ret_dict["{}/{}".format(k, ka)] = atr

    return ret_dict


def ds2dict(ds: xr.Dataset):
    """
    Convert all Dataset coordinates data and coordinates variables and all Dataset attributes into single dictionary.

    :param ds:
    :return: (dict) Dataset converted into single OrderedDict.
    """

    ret_dict = dict()
    ret_dict.update(ds.attrs)

    # Add all variables:
    for k, val in ds.variables.items():
        ret_dict[k] = val.data
        for ka, atr in val.attrs.items():
            ret_dict["{}/{}".format(k, ka)] = atr

    return ret_dict




def dict2xr(dictionary: dict):
    pass
