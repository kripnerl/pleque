import xarray as xr
import numpy as np
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

    def get_time_slice(self, time: float, tolerance=None):
        """
        Creates an Equilibrium from the slice nearest to the specified time

        :param time: float, time in ms
        :param tolerance: float or None, raise ValueError if the selected time slice is outside the tolerance.
                          If None the warning is shown if the time difference is more then 10 ms.
        """
        ds = self.eqs_dataset.dropna(dim='time', how='all') \
            .sel(time=time, method='nearest') \
            .rename({'Rt': 'R', 'Zt': 'Z'})
        if tolerance is not None and np.abs(ds.time - time) > tolerance:
            raise ValueError('Insufficient time slice found! Delta time: {:.1f} ms\n'
                             '         Required time: {:.1f} ms, selected time {:.1f} ms. '
                             .format(ds.time.item() - time, time, ds.time.item()))

        elif np.abs(ds.time - time) > 10:
            print('!!!!!!!!!!!')
            print('WARNING: Insufficient time slice found! Delta time: {:.1f} ms\n'
                  '         Required time: {:.1f} ms, selected time {:.1f} ms. '
                  .format(ds.time.item() - time, time, ds.time.item()))
            print('!!!!!!!!!!!')
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
        ret_dict["data"] = da.values
    else:
        ret_dict[da.name] = da.values

    # Add axes:
    for k, val in da.coords.items():
        ret_dict[k] = val.values
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
        ret_dict[k] = val.values
        for ka, atr in val.attrs.items():
            ret_dict["{}/{}".format(k, ka)] = atr

    return ret_dict




def dict2xr(dictionary: dict):
    pass
