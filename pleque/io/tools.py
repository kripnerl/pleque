import xarray as xr
from collections import OrderedDict


def xr2dict(ds: xr.Dataset):
    """
    Convert all Dataset coordinates data and coordinates variables and all Dataset attributes into single dictionary.

    :param ds:
    :return: (OrderedDict) Dataset converted into single OrderedDict.
    """

    ret_dict = OrderedDict()
    ret_dict.update(ds.variables)
    ret_dict.update(ds.attrs)

    return ret_dict
