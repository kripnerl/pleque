#from collections.abc import Sequence
#import itertools

import numpy as np
#import xarray

#from pleque.utils.decorators import deprecated


class FluxFunction:
    # def interpolate(self, coords, data)
    # def interpolate(self, R, Z, data):
    #     pass

    def __init__(self, equi):
        # _flux_funcs = ['psi', 'rho']
        _flux_funcs = ['psi_n', 'psi', 'rho']
        _coordinates_funcs = ['coordinates']
        self._equi = equi
        self._func_names = []
        # self.__dict__.update(_flux_funcs)  # at class level?
        for fn in _flux_funcs:
            setattr(self, fn, getattr(self._equi, fn))  # methods are bound to _equi
        for fn in _coordinates_funcs:
            setattr(self, fn, getattr(self._equi, fn))  # methods are bound to _equi

    def add_flux_func(self, name, data, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, spline_smooth=0,
                      spline_order=3, **coords):

        from scipy.interpolate import UnivariateSpline

        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)
        # interp = interpolate(psi_n, data)
        # todo: unique
        # psi_n = coord.psi_n
        # idxs = np.argsort(psi_n)
        # psi_n = psi_n[idxs]
        # data = data[idxs]
        psi_n, idxs = np.unique(coord.psi_n, return_index=True)
        data = data[idxs]

        # add flux function name to the list
        self._func_names.append(name)

        interp = UnivariateSpline(psi_n, data, s=spline_smooth, k=spline_order)
        setattr(self, '_interp_' + name, interp)

        def new_func(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, **coords):
            coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)
            return interp(coord.psi_n)

        setattr(type(self), name, new_func)

    def __getitem__(self, item):
        return getattr(self, item)

    def keys(self):
        return self._func_names
