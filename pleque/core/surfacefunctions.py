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
        indr, indz = self.evalcoord(coord)
        print(coord.R[indr].shape)
        print(coord.Z[indz].shape)
        data_= data[indr, :]
        data = data[:, indz]
        print(data.shape)
        self._func_names.append(name)

        interp2d = RectBivariateSpline(coord.R[indr], coord.Z[indz], data, kx=spline_order, ky=spline_order,
                                       s=spline_smooth)
        setattr(self, '_interp2D_' + name, interp2d)

        def new_func(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, **coords):
            coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)
            return interp2d(coord.R, coord.Z)

        setattr(self, name, types.MethodType(new_func, self))

        return interp2d

    @staticmethod
    def evalcoord(coord):
        """
        evaluate if R, Z are increasing and unique and have the same size
        :return: indexes of R, Z
        """
        indr = np.where(np.diff(coord.R) > 0)[0]
        indz = np.where(np.diff(coord.Z) > 0)[0]
        if len(indz) < 1:
            raise Exception('Z coordinates should not decrease. Flip Z coordinate')
        if len(indr) < 1:
            raise Exception('R coordinates should not decrease., Flip the R coordinate')
        if len(indr) != len(indz):
            raise Exception('selected R, Z do not have the same dimensions!')
        else:
            indr = np.concatenate((indr, [len(coord.R) - 1]), axis=0)
            indz = np.concatenate((indz, [len(coord.Z) - 1]), axis=0)
        return indr, indz

    # R = np.linspace(0.3, 0.4, 10)
    # Z = np.linspace(0, 0.2, 10)
    # coord = eq.coordinates(R, Z)
    # r, z = evalcoord(coord)

    def keys(self):
        return self._func_names
