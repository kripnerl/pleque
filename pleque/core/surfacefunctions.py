import numpy as np


class SurfaceFunctions():
    """
    2D surface function
    input equilibrium, name, data, coordinates
    """

    def __init__(self, equi):
        _coordinates_funcs = ['coordinates']
        self._equi = equi
        for fn in _coordinates_funcs:
            setattr(self, fn, getattr(self._equi, fn))  # methods are bound to _equi

    def add_surface_func(self, name, data, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, spline_smooth=0,
                         spline_order=3, **coords):

        from scipy.interpolate import BivariateSpline
        # from scipy.integrate import RectBivariateSpline

        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)
        psi_n, idxs = np.unique(coord.psi_n, return_index=True)
        data = data[idxs]

        # add flux function name to the list
        self._func_names.append(name)

        interp = BivariateSpline(psi_n, data, s=spline_smooth, k=spline_order)
        setattr(self, '_interp_' + name, interp)

        def new_func(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, **coords):
            coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)
            return interp(coord.psi_n)

        setattr(self, name, types.MethodType(new_func, self))

        return interp
