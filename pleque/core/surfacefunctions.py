import types
import numpy as np


class SurfaceFunctions:
    """
    2D surface function
    input equilibrium, name, data, coordinates
    """

    def __init__(self, equi):
        _flux_funcs = ['psi_n', 'psi', 'rho']
        _coordinates_funcs = ['coordinates']
        self._equi = equi
        self._func_names = []
        for fn in _flux_funcs:
            setattr(self, fn, getattr(self._equi, fn))  # methods are bound to _equi
        for fn in _coordinates_funcs:
            setattr(self, fn, getattr(self._equi, fn))  # methods are bound to _equi

    def add_surface_func(self, name, data, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, spline_smooth=0,
                         spline_order=3, **coords):

        from scipy.interpolate import RectBivariateSpline

        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)
        r, z = self.evalcoord(coord)
        data = data(r, z)
        self._func_names.append(name)

        interp2d = RectBivariateSpline(r, z, data, kx=spline_order, ky=spline_order, s=spline_smooth)
        setattr(self, '_interp2D_' + name, interp2d)

        def new_func(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, **coords):
            coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)
            return interp2d(coord.R, coord.Z)

        setattr(self, name, types.MethodType(new_func, self))

        return interp2d

    @staticmethod
    def evalcoord(coord):
        """
        evaluate if R, Z are increasing and have the same dimension
        :return: R, Z with the same size
        """
        import warnings
        indr = np.where(np.diff(coord.R) > 0)[0]
        indz = np.where(np.diff(coord.Z) > 0)[0]
        if len(indr) > 1 and len(indz) > 1:
            r = coord.R[indr]
            z = coord.Z[indz]
        if len(indz) < 1:
            # z = np.flip(coord.Z)
            warnings.warn('Z coordinates should not decrease.')
            # raise Exception('Z coordinates should not decrease., Z values were flipped')
        if len(indr) < 1:
            # r = np.flip(coord.R)
            warnings.warn('R coordinates should not decrease.')
            # raise Exception('R coordinates should not decrease., R values were flipped')
        if len(indr) > len(indz):
            z = np.linspace(min(coord.Z), max(coord.Z), len(indr))
            warnings.warn('size of R coordinates is not equal to the size of Z coordinates. Now they have same size')
            # raise Exception(
            #    'size of R coordinates is not equal to the size of Z coordinates. \
            #    Z coordinates now has the same size as R (which was larger)')
        elif len(indr) < len(indz):
            r = np.linspace(min(coord.R), max(coord.R), len(indz))
            warnings.warn('size of R coordinates is not equal to the size of Z coordinates. Now they have same size')
            # raise Exception(
            #    'size of Z coordinates is not equal to the size of R coordinates. \
            #    R coordinates now has the same size as Z (which was larger)')
        return r, z

    def keys(self):
        return self._func_names
