import xarray as xr
from collections import OrderedDict
from typing import Union


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
