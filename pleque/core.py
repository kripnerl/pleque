from collections import Sequence

import numpy as np
import xarray


class FluxFuncs:
    # def interpolate(self, coords, data)
    # def interpolate(self, R, Z, data):
    #     pass

    def __init__(self, equi):
        # _flux_funcs = ['psi', 'rho']
        _flux_funcs = ['psi_n', 'psi', 'rho']
        _coordinates_funcs = ['coordinates']
        self._equi = equi
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

        interp = UnivariateSpline(psi_n, data, s=spline_smooth, k=spline_order)
        setattr(self, '_interp_' + name, interp)

        def new_func(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, **coords):
            coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)
            return interp(coord.psi_n)

        setattr(type(self), name, new_func)


class Equilibrium(object):
    """
    Equilibrium class ...
    """

    # def __init__(self,
    #              basedata: xarray.Dataset,
    #              first_wall=None: Iterable[(float, float)],
    #              psi_lcfs=None: float,
    #              X_points=None: Iterable[(float, float)],
    #              strike_points=None: Iterable[(float, float)],
    #              spline_order=5: int,
    #              cocos=13: int,
    #             ):
    def __init__(self,
                 basedata: xarray.Dataset,
                 first_wall=None,
                 psi_lcfs=None,
                 x_points=None,
                 strike_points=None,
                 spline_order=3,
                 spline_smooth=0,
                 cocos=-1,
                 verbose=True
                 ):
        """
        Equilibrium class instance should be obtained generally by functions in pleque.io
        package.

        Optional arguments may help the initialization.
    
        Arguments
        ---------
        basedata: xarray.Dataset with psi(R, Z) on a rectangular R, Z grid, f(psi_norm), p(psi_norm)
        first_wall: required for initialization in case of limiter configuration
        """
        from scipy.interpolate import RectBivariateSpline, UnivariateSpline

        if verbose:
            print('---------------------------------')
            print('Equilibrium module initialization')
            print('---------------------------------')

        self._basedata = basedata
        self._verbose = verbose
        self._first_wall = first_wall
        self._psi_lcfs = psi_lcfs
        self._x_points = x_points
        self._strike_point = strike_points
        self._spline_order = spline_order
        self._cocos = cocos

        # todo: resolve this from input
        self._Bpol_sign = 1

        r = basedata.R.data
        z = basedata.Z.data
        psi = basedata.psi.transpose('R', 'Z').data

        self.R_min = np.min(r)
        self.R_max = np.max(r)
        self.Z_min = np.min(z)
        self.Z_max = np.max(z)

        if verbose:
            print('--- Generate 2D spline ---')

        # generate spline:
        # todo: first assume r, z are ascending:
        spl = RectBivariateSpline(r, z, psi, kx=spline_order, ky=spline_order,
                                  s=spline_smooth)
        self._spl_psi = spl

        if verbose:
            print('--- Looking for extremes ---')
        # find extremes:
        self.__find_extremes__()

        # generate 1d profiles:
        if self._psi_lcfs - self._psi_axis > 0:
            self._psi_sign = +1
        else:
            self._psi_sign = -1

        psi_n = basedata.psi_n.data
        pressure = basedata.pressure.data
        fpol = basedata.fpol.data
        self.BvacR = fpol[-1]

        if verbose:
            print('--- Generate 1D splines ---')

        if verbose:
            print('--- Mapping midplane to psi_n ---')

        self.__map_midplane2psi__()

        if verbose:
            print('--- Mapping pressure and f func to psi_n ---')

        self._fpol_spl = UnivariateSpline(psi_n, fpol, k=3, s=0)
        self._df_dpsin_spl = self._fpol_spl.derivative()
        self._pressure_spl = UnivariateSpline(psi_n, pressure, k=3, s=0)
        self._dp_dpsin_spl = self._pressure_spl.derivative()

        self.fluxfuncs.add_flux_func('fpol', fpol, psi_n=psi_n)
        self.fluxfuncs.add_flux_func('pressure', pressure, psi_n=psi_n)

    def psi(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        """
        Psi value

        :param psi_n:
        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return coord.psi

    def psi_n(self, *coordinates, R=None, Z=None, psi=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi=psi, coord_type=coord_type, grid=grid, **coords)
        return coord.psi_n

    @property
    def _diff_psi_n(self):
        return 1 / (self._psi_lcfs - self._psi_axis)

    def rho(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return np.sqrt(coord.psi_n)

    def pressure(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._pressure_spl(coord.psi_n)

    def pprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._dp_dpsin_spl(coord.psi_n) * self._diff_psi_n

    def fpol(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        # todo use in_plasma
        mask_out = coord.psi_n > 1
        fpol = self._fpol_spl(coord.psi_n)
        fpol[mask_out] = self.BvacR
        return fpol

    def fprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        mask_out = coord.psi_n > 1
        fprime = self._df_dpsin_spl(coord.psi_n) * self._diff_psi_n
        fprime[mask_out] = 0
        return fprime

    def ffprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        mask_out = coord.psi_n > 1
        ffprime = self._fpol_spl(coord.psi_n) * self._df_dpsin_spl(coord.psi_n) * self._diff_psi_n
        ffprime[mask_out] = 0
        return ffprime

    def q(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def diff_q(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def tor_flux(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def B_abs(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Absolute value of magnetic field in Tesla.

        :param grid:
        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return: Absolute value of magnetic field in Tesla.
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        B_R = self.B_R(coord)
        B_Z = self.B_Z(coord)
        B_T = self.B_tor(coord)
        B_abs = np.sqrt(B_R ** 2 + B_Z ** 2 + B_T ** 2)

        return B_abs

    def flux_surface(self, *coordinates, resolution=[1e-3, 1e-3], dim="step",
                     closed=True, inlcfs=True, R=None, Z=None, psi_n=None,
                     coord_type=None, **coords):
        """
        Function which finds flux surfaces with requested values of psi or psi-normalized. Specification of the
        fluxsurface properties as if it is inside last closed flux surface or if the surface is supposed to be
        closed are possible.
        :param R:
        :param Z:
        :param psi_n:
        :param coord_type:
        :param coordinates: specifies flux surface to search for (by spatial point or values of psi or psi normalised).
        If coordinates is spatial point (dim=2) then parameters closed and lcfs are automatically overridden.
        Coordinates.grid must be False.
        :param resolution: Iterable of size 2 or a number, default is [1e-3, 1e-3]. If a number is passed,
         R and Z dimensions will have the same size or step (depending on dim parameter). Different R and Z
          resolutions or dimension sizes can be required by passing an iterable of size 2
        :param dim: iterable of size 2 or string. Default is "step", determines the meaning of the resolution.
         If "step" used, values in resolution are interpreted as step length in psi poloidal map. If "size" is used,
        values in resolution are interpreted as requested number of points in a dimension. If string is passed,
         same value is used for R and Z dimension. Different interpretation of resolution for R, Z dimensions can be
         achieved by passing an iterable of shape 2.
        :param closed: Are we looking for a closed surface. This parameter is ignored of inlcfs is True.
        :param inlcfs: If True only the surface inside the last closed flux surface is returned.
        :return: list of FluxSurface objects
        """

        from pleque.fluxsurface import FluxSurface

        coordinates = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)

        # get the grid for psi map to find the contour in.
        grid = self.grid(resolution=resolution, dim=dim)

        # create coordinates
        # coords = self.coordinates(R=R, Z=Z, grid=True, coord_type=["R", "Z"])

        # get the coordinates of the contours with requested leve and convert them into
        # instances of FluxSurface class
        contour = self._get_surface(grid, level=coordinates.psi_n[0], norm=True)

        for i in range(len(contour)):
            contour[i] = FluxSurface(contour[i])

        # get the position of the magnetic axis, which is used to determine whether the found fluxsurfaces are
        # within the lcfs
        magaxis = self.coordinates(np.expand_dims(self._mg_axis, axis=0))

        # find fluxsurfaces with requested parameters
        fluxsurface = []

        if coordinates.dim == 1:
            for i in range(len(contour)):
                if inlcfs and contour[i].closed and contour[i].contains(magaxis):
                    fluxsurface.append(contour[i])
                    return fluxsurface
                elif not inlcfs and closed and contour[i].closed:
                    fluxsurface.append(contour[i])
                elif not inlcfs and not closed and not contour[i].closed:
                    fluxsurface.append(contour[i])
        elif coordinates.dim == 2:
            # Sadly contour des not go through the point due to mesh resolution :-(
            dist = np.inf
            tmp2 = None
            
            for i in range(len(contour)):
                tmp = contour[i].distance(coordinates)
                if tmp < dist:
                    dist = tmp
                    tmp2 = contour[i]

            fluxsurface.append(tmp2)

        return fluxsurface

    def _get_surface(self, *coordinates, R=None, Z=None, level=0.5, norm=True, coord_type=None, **coords):
        """
        finds contours
        :return: list of coordinates of contours on a requested level
        """
        from pleque.utils.surfaces import find_contour

        coordinates = self.coordinates(*coordinates, R=R, Z=Z, grid=True, coord_type=coord_type, **coords)

        if norm:
            contour = find_contour(coordinates.psi_n, level=level, r=coordinates.R, z=coordinates.Z)
        else:
            contour = find_contour(coordinates.psi, level=level, r=coordinates.R, z=coordinates.Z)

        for i in range(len(contour)):
            contour[i] = Coordinates(self, contour[i])

        return contour

    def grid(self, resolution=None, dim="step"):
        """
        Function which returns 2d grid with requested step/dimensions generated over the reconstruction space.
        :param resolution: Iterable of size 2 or a number, default is [1e-3, 1e-3]. If a number is passed,
         R and Z dimensions will have the same size or step (depending on dim parameter). Different R and Z
          resolutions or dimension sizes can be required by passing an iterable of size 2.
          If None, default grid is returned.
        :param dim: iterable of size 2 or string ('step', 'size'). Default is "step", determines the meaning
            of the resolution.
         If "step" used, values in resolution are interpreted as step length in psi poloidal map. If "size" is used,
        values in resolution are interpreted as requested number of points in a dimension. If string is passed,
         same value is used for R and Z dimension. Different interpretation of resolution for R, Z dimensions can be
         achieved by passing an iterable of shape 2.
        :return: Instance of `Coordinates` class with grid data
        """
        if resolution is None:
            R = self._basedata.R.data
            Z = self._basedata.Z.data
        else:
            if isinstance(resolution, Sequence):
                if not len(resolution) == 2:
                    raise ValueError("if iterable, resolution has to be of size 2")
                res_R = resolution[0]
                res_Z = resolution[1]
            else:
                res_R = resolution
                res_Z = resolution

            if isinstance(dim, Sequence) and len(dim) == 2:
                if res_R is None:
                    R = self._basedata.R.data
                elif dim[0] == "step":
                    R = np.arange(self._basedata.R.min(), self._basedata.R.max(), res_R)
                elif dim[0] == "size":
                    R = np.linspace(self._basedata.R.min(), self._basedata.R.max(), res_Z)
                else:
                    raise ValueError("Wrong dim[0] value passed")

                if res_Z is None:
                    Z = self._basedata.Z.data
                elif dim[1] == "step":
                    Z = np.arange(self._basedata.Z.min(), self._basedata.R.max(), res_R)
                elif dim[1] == "size":
                    Z = np.linspace(self._basedata.Z.min(), self._basedata.Z.max(), res_Z)
                else:
                    raise ValueError("Wrong dim[1] value passed")
            elif isinstance(dim, str):
                if dim == "step":
                    if res_R is None:
                        R = self._basedata.R.data
                    else:
                        R = np.arange(self._basedata.R.min(), self._basedata.R.max(), res_R)
                    if res_Z is None:
                        Z = self._basedata.Z.data
                    else:
                        Z = np.arange(self._basedata.Z.min(), self._basedata.Z.max(), res_Z)
                elif dim == "size":
                    if res_R is None:
                        R = self._basedata.R.data
                    else:
                        R = np.linspace(self._basedata.R.min(), self._basedata.R.max(), res_R)
                    if res_Z is None:
                        Z = self._basedata.Z.data
                    else:
                        Z = np.linspace(self._basedata.Z.min(), self._basedata.Z.max(), res_Z)
                else:
                    raise ValueError("Wrong dim value passed")
            else:
                raise ValueError("Wrong dim value passed")

        coords = self.coordinates(R=R, Z=Z, grid=True)

        return coords

    # todo: resolve the grids
    def B_R(self, *coordinates, R=None, Z=None, coord_type=('R', 'Z'), grid=True, **coords):
        """
        Poloidal value of magnetic field in Tesla.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return -self._spl_psi(coord.R, coord.Z, dy=1, grid=coord.grid).T / coord.R * self._Bpol_sign

    def B_Z(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Poloidal value of magnetic field in Tesla.

        :param grid:
        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return self._spl_psi(coord.R, coord.Z, dx=1, grid=coord.grid).T / coord.R * self._Bpol_sign

    def B_pol(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Absolute value of magnetic field in Tesla.

        :param grid:
        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        B_R = self.B_R(coord)
        B_Z = self.B_Z(coord)
        B_pol = np.sqrt(B_R ** 2 + B_Z ** 2)
        return B_pol

    def B_tor(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        """
        Toroidal value of magnetic field in Tesla.

        :param grid:
        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return self.fpol(coord) / coord.R

    def q(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def tor_flux(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def diff_q(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def j_R(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def j_Z(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def j_pol(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    def j_tor(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        raise NotImplementedError("This method hasn't been implemented yet. "
                                  "Use monkey patching in the specific cases.")

    @property
    def lcfs(self):
        return self.coordinates(self._lcfs)

    @property
    def first_wall(self):
        return self.coordinates(self._first_wall)

    @property
    def magnetic_axis(self):
        return self.coordinates(self._mg_axis[0], self._mg_axis[1])

    def coordinates(self, *coordinates, coord_type=None, grid=False, **coords):
        """
        Return instance of Coordinates. If instances of coordinates is already on the input, just pass it throught.
        :param coordinates:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        if len(coordinates) >= 1 and isinstance(coordinates[0], Coordinates):
            return coordinates[0]
        else:
            return Coordinates(self, *coordinates, coord_type=coord_type, grid=grid, **coords)

    def in_first_wall(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        from pleque.utils.surfaces import point_inside_curve
        # if grid:
        #     r_mesh, z_mesh = np.meshgrid(R, Z)
        #     points = np.vstack((r_mesh.ravel(), z_mesh.ravel())).T
        # else:
        #     points = np.vstack((R, Z)).T
        # mask_in = point_in_first_wall(self, points)
        # return mask_in
        points = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)

        mask_in = point_inside_curve(points.as_array(), self._first_wall)
        if points.grid:
            mask_in = mask_in.reshape(len(points.x2), len(points.x1))
        return mask_in

    def in_lcfs(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        from pleque.utils.surfaces import point_inside_curve
        # if grid:
        #     r_mesh, z_mesh = np.meshgrid(R, Z)
        #     points = np.vstack((r_mesh.ravel(), z_mesh.ravel())).T
        # else:
        #     points = np.vstack((R, Z)).T
        points = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)

        mask_in = point_inside_curve(points.as_array(), self._lcfs)
        if points.grid:
            mask_in = mask_in.reshape(len(points.x2), len(points.x1))
        return mask_in

    def __find_extremes__(self):
        from scipy.signal import argrelmin
        from scipy.optimize import minimize

        # for sure not the best algorithm ever...
        rs = np.linspace(self.R_min, self.R_max, 300)
        zs = np.linspace(self.Z_min, self.Z_max, 400)

        psi = self._spl_psi(rs, zs)
        psi_x = self._spl_psi(rs, zs, dx=1, dy=0)
        psi_y = self._spl_psi(rs, zs, dx=0, dy=1)
        psi_xysq = psi_x ** 2 + psi_y ** 2

        mins0 = tuple(argrelmin(psi_xysq, axis=0))
        mins1 = tuple(argrelmin(psi_xysq, axis=1))

        import matplotlib.pyplot as plt
        # plt.figure()
        # plt.pcolormesh(rs, zs, psi.T)
        # plt.plot(mins0[0], mins0[1])
        # plt.contour(rs, zs, psi_xy.T, [0], colors='C4', ls='--')
        # plt.show()

        def psi_xysq_func(x):
            return self._spl_psi(x[0], x[1], dx=1, dy=0, grid=False) ** 2 \
                   + self._spl_psi(x[0], x[1], dx=0, dy=1, grid=False) ** 2

        x_points = []
        o_points = []

        for i, (ar, az) in enumerate(zip(mins0[0], mins0[1])):
            for j, (br, bz) in enumerate(zip(mins1[0], mins1[1])):
                if ar == br and az == bz:
                    r_ex = rs[ar]
                    z_ex = zs[az]
                    x0 = np.array((r_ex, z_ex))

                    # minimize in the vicinity:
                    bounds = ((np.max((self.R_min, r_ex - 0.1)),
                               np.min((self.R_max, r_ex + 0.1))),
                              (np.max((self.Z_min, z_ex - 0.1)),
                               np.min((self.Z_max, z_ex + 0.1))))

                    res = minimize(psi_xysq_func, x0, bounds=bounds)
                    # Remove bad candidates for extreme
                    if res['fun'] > 1e-3:
                        continue
                    r_ex2 = res['x'][0]
                    z_ex2 = res['x'][1]

                    #                    psi_xyabs = np.abs(psi_xy[ar, az])
                    psi_xy = (self._spl_psi(r_ex2, z_ex2, dx=1, dy=1, grid=False)) ** 2
                    psi_xx = (self._spl_psi(r_ex2, z_ex2, dx=2, dy=0, grid=False))
                    psi_yy = (self._spl_psi(r_ex2, z_ex2, dx=0, dy=2, grid=False))
                    D = psi_xx * psi_yy - psi_xy

                    if D > 0:
                        # plt.plot(rs[ar], zs[az], 'o', markersize=10, color='b')
                        # plt.plot(r_ex2, z_ex2, 'o', markersize=8, color='C4')
                        o_points.append((r_ex2, z_ex2))
                    else:
                        # plt.plot(rs[ar], zs[az], 'x', markersize=10, color='r')
                        # plt.plot(r_ex2, z_ex2, 'x', markersize=8, color='C5')
                        x_points.append((r_ex2, z_ex2))

        # todo: After beeing function written, check whether are points inside limiter

        # First identify the o-point nearest the operation range as center of plasma
        r_centr = (self.R_min + self.R_max) / 2
        z_centr = (self.Z_min + self.Z_max) / 2
        o_points = np.array(o_points)
        x_points = np.array(x_points)

        op_dist = (o_points[:, 0] - r_centr) ** 2 + (o_points[:, 1] - z_centr) ** 2
        # assume that psi value has its minimum in the center
        op_psiscale = self._spl_psi(o_points[:, 0], o_points[:, 1], grid=False)
        op_psiscale = 1 + (op_psiscale - np.min(op_psiscale)) / (np.max(op_psiscale) - np.min(op_psiscale))

        op_in_first_wall = np.ones_like(op_dist)
        if self._first_wall is not None:
            op_in_first_wall = self.in_first_wall(R=o_points[:, 0],
                                                  Z=o_points[:, 1],
                                                  grid=False) * 1
            op_in_first_wall = np.abs(op_in_first_wall - 1 + 1e-3)

        sortidx = np.argsort(op_dist * op_psiscale * op_in_first_wall)
        # idx = np.argmin(op_dist)
        self._mg_axis = o_points[sortidx[0]]
        self._psi_axis = np.asscalar(self._spl_psi(self._mg_axis[0], self._mg_axis[1]))
        self._opoints = o_points[sortidx]

        # identify THE x-point as the x-point nearest in psi value to mg_axis
        # todo: Ensure that the psi function between x-point and o-point is monotonic (!)

        psi_diff = np.zeros(x_points.shape[0])
        for i in np.arange(x_points.shape[0]):
            rxp = x_points[i, 0]
            zxp = x_points[i, 1]
            psi_xp = np.asscalar(self._spl_psi(rxp, zxp))
            psi_diff[i] = np.abs(psi_xp - self._psi_axis)

        xp_dist = (x_points[:, 0] - self._mg_axis[0]) ** 2 + (x_points[:, 1] - self._mg_axis[1]) ** 2
        xp_dist = (xp_dist - np.min(xp_dist)) / (np.max(xp_dist) - np.min(xp_dist))

        # idx = np.argmin(psi_diff)
        sortidx = np.argsort(psi_diff * xp_dist)

        self._x_point = x_points[sortidx[0]]
        self._psi_lcfs = np.asscalar(self._spl_psi(self._x_point[0], self._x_point[1]))
        self._x_points = x_points[sortidx]

        self._x_point2 = x_points[sortidx[1]]
        self._psi_xp2 = np.asscalar(self._spl_psi(self._x_point2[0], self._x_point2[1]))

        # get lcfs, for now using matplotlib contour line

        # todo: replace this by Matisek's function
        plt.figure(1111)
        cl = plt.contour(rs, zs, psi.T, [self._psi_lcfs])
        paths = cl.collections[0].get_paths()
        v = np.concatenate([p.vertices for p in paths], axis=0)
        plt.close(1111)

        if self._x_point[1] < self._x_point2[1]:
            if self._verbose:
                print('>>> lower x-point configuration found')
            v = v[v[:, 1] > self._x_point[1], :]
            v = v[v[:, 1] < self._x_point2[1], :]

        else:
            if self._verbose:
                print('>>> upper x-point configuration found')
            v = v[v[:, 1] < self._x_point[1], :]
            v = v[v[:, 1] > self._x_point2[1], :]

        mask_in = self.in_first_wall(R=v[:, 0], Z=v[:, 1], grid=False)
        v = v[mask_in, :]

        self._lcfs = v

    @property
    def fluxfuncs(self):
        return FluxFuncs(self)  # filters out methods from self

    def __map_midplane2psi__(self):
        from scipy.interpolate import UnivariateSpline

        r_mid = np.linspace(0, self.R_max - self._mg_axis[0], 100)
        psi_mid = self.psi(r_mid + self._mg_axis[0], self._mg_axis[1] * np.ones_like(r_mid), grid=False)

        from .utils.tools import arglis

        if self._psi_axis < self._psi_lcfs:
            # psi increasing:
            idxs = arglis(psi_mid)
        else:
            # psi decreasing
            idxs = arglis(psi_mid[::-1])
            idxs = idxs[::-1]

        psi_mid = psi_mid[idxs]
        r_mid = r_mid[idxs]
        self._rmid_spl = UnivariateSpline(psi_mid, r_mid, k=3, s=0)


class Coordinates(object):

    def __init__(self, equilibrium: Equilibrium, *coordinates, coord_type=None, grid=False, **coords):
        self._eq = equilibrium
        self._valid_coordinates = {'R', 'Z', 'psi_n', 'psi', 'rho', 'r', 'theta', 'phi', 'X', 'Y'}
        self._valid_coordinates_1d = {('psi_n',), ('psi',), ('rho',)}
        self._valid_coordinates_2d = {('R', 'Z'), ('r', 'theta')}
        self._valid_coordinates_3d = {('R', 'Z', 'phi'), ('X', 'Y', 'Z')}
        self.dim = -1  # init only
        self.grid = grid

        self.__evaluate_input__(*coordinates, coord_type=coord_type, **coords)

    # def __call__(self, *args, **kwargs):
    #     pass
    #
    # def __iter__(self):
    #     pass

    def __len__(self):
        if self.grid:
            return len(self.x1) * len(self.x2)
        else:
            return len(self.x1)

    # def sort(self, order):
    #     pass

    @property
    def R(self):
        if self.dim >= 2:
            return self.x1

    @property
    def Z(self):
        if self.dim >= 2:
            return self.x2

    @property
    def psi(self):
        if self.dim == 1:
            return self._eq._psi_axis + self.x1 * (self._eq._psi_lcfs - self._eq._psi_axis)
        elif self.dim >= 2:
            psi = self._eq._spl_psi(self.x1, self.x2, grid=self.grid)
            if self.grid:
                return psi.T
            else:
                return psi

    @property
    def psi_n(self):
        if self.dim == 1:
            return self.x1
        elif self.dim >= 2:
            return (self.psi - self._eq._psi_axis) / (self._eq._psi_lcfs - self._eq._psi_axis)

    @property
    def rho(self):
        return np.sqrt(self.psi_n)

    @property
    def r(self):
        r_mgax, z_mgax = self._eq._mg_axis
        return np.sqrt((self.x1 - r_mgax) ** 2 + (self.x2 - z_mgax) ** 2)

    @property
    def theta(self):
        r_mgax, z_mgax = self._eq._mg_axis
        return np.arctan2((self.x2 - z_mgax), (self.x1 - r_mgax))

    @property
    def r_mid(self):
        return self._eq._rmid_spl(self.psi)

    @property
    def phi(self):
        return self.x3
    
    @property
    def X(self):
        if self.dim >= 2:
            return self.R * np.cos(self.phi)

    @property
    def Y(self):
        if self.dim >= 2:
            return self.R * np.sin(self.phi)

    def mesh(self):
        if self.dim != 2 or not self.grid:
            raise TypeError('mesh can be returned only for 2d grid coordinates.')
        return np.meshgrid(self.x1, self.x2)

    # todo
    # @property
    # def r_mid(self):
    #     """
    #     Midplane coordinate.
    #     :return:
    #     """
    #
    #
    #     return

    def as_array(self, coord_type=None):
        """
        Return array of size (N, dim), where N is number of points and dim number of dimensions specified by coord_type
        :param coord_type: not effected at the moment (TODO)
        :return:
        """
        if self.dim == 0:
            return np.array(())
        # coord_type_ = self._verify_coord_type(coord_type)
        elif self.dim == 1:
            return self.x1 
        elif self.dim == 2:
            if self.grid:
                x1, x2 = self.mesh()
                return np.vstack((x1.ravel(), x2.ravel())).T
                # x1 = x1.ravel()
                # x2 = x2.ravel()
                # return np.array([x1, x2]).T
            else:
                return np.array([self.x1, self.x2]).T
        elif self.dim == 3:
            # todo: replace this by split method
            return np.array([self.x1, self.x2, self.x3]).T

    def __evaluate_input__(self, *coordinates, coord_type=None, **coords):
        from collections import Iterable

        if len(coordinates) == 0:
            # todo:
            self.dim = 0
            xy = []
            xy_name = []
            for key, val in coords.items():
                if key in self._valid_coordinates and val is not None:
                    if isinstance(val, Iterable):
                        if not isinstance(val, np.ndarray):
                            val = np.array(val, ndmin=1)
                        if len(val.shape) == 0:
                            val = val.reshape((len(val), 1))
                    else:
                        val = np.array(val, ndmin=1)
                    xy.append(val)
                    xy_name.append(key)
                    self.dim += 1

            if self.dim == 0:
                coord_type_ = ()
            elif self.dim == 1:
                self._x1_input = xy[0]
                coord_type_ = tuple(xy_name)
            elif self.dim == 2:
                if tuple(xy_name) in self._valid_coordinates_2d:
                    self._x1_input = xy[0]
                    self._x2_input = xy[1]
                    coord_type_ = tuple(xy_name)
                elif tuple(xy_name[::-1]) in self._valid_coordinates_2d:
                    self._x1_input = xy[1]
                    self._x2_input = xy[0]
                    coord_type_ = tuple(xy_name[::-1])
                    if len(self._x1_input) != len(self._x2_input) and not self.grid:
                        raise ValueError('All coordinates should contain same dimension.')
                else:
                    raise ValueError('Invalid combination of input coordinates.')
            elif self.dim == 3:
                if tuple(xy_name) in self._valid_coordinates_3d:
                    # todo: implement various order of coordinates
                    self._x1_input = xy[0]
                    self._x2_input = xy[1]
                    self._x3_input = xy[2]
                    coord_type_ = tuple(xy_name)

            else:
                # self._incompatible_dimension_error(self.dim)
                raise ValueError('Operation in {} space has not be en allowed yet. Sorry.'
                                 .format(self.dim))

            self._coord_type_input = coord_type_
        else:
            if len(coordinates) == 1:
                xy = coordinates[0]

                if self.grid:
                    print('WARNING: grid == True is not allowed for this coordinates input. '
                          'Turning grid = False.')
                self.grid = False

                if isinstance(xy, Iterable):
                    # input as array of size (N, dim),
                    # if (N) add dimension
                    if not isinstance(xy, np.ndarray):
                        xy = np.array(xy, ndmin=2)
                    if len(xy.shape) == 1:
                        xy = xy.reshape((len(xy), 1))

                    self.dim = xy.shape[1]
                    if self.dim == 1:
                        self._x1_input = xy[:, 0]
                    elif self.dim == 2:
                        self._x1_input = xy[:, 0]
                        self._x2_input = xy[:, 1]
                    elif self.dim == 3:
                        self._x1_input = xy[:, 0]
                        self._x2_input = xy[:, 1]
                        self._x3_input = xy[:, 2]
                    else:
                        self._incompatible_dimension_error(self.dim)
                else:
                    # 1d, one number
                    self.dim = 1
                    self._x1_input = np.array([xy])
            elif len(coordinates) == 2:
                self.dim = 2
                x1 = coordinates[0]
                x2 = coordinates[1]

                # assume _x1_input and _x2_input to be arrays of size (N)
                if not isinstance(x1, np.ndarray):
                    x1 = np.array(x1, ndmin=1)
                if not isinstance(x2, np.ndarray):
                    x2 = np.array(x2, ndmin=1)
                self._x1_input = x1
                self._x2_input = x2
            elif len(coordinates) == 3:
                self.dim = 3
                x1 = coordinates[0]
                x2 = coordinates[1]
                x3 = coordinates[2]

                # assume _x1_input and _x2_input to be arrays of size (N)
                if not isinstance(x1, np.ndarray):
                    x1 = np.array(x1, ndmin=1)
                if not isinstance(x2, np.ndarray):
                    x2 = np.array(x2, ndmin=1)
                if not isinstance(x3, np.ndarray):
                    x3 = np.array(x3, ndmin=1)
                self._x1_input = x1
                self._x2_input = x2
                self._x3_input = x3

            else:
                self._incompatible_dimension_error(len(coordinates))

            self._coord_type_input = self._verify_coord_type(coord_type)

        self._convert_to_default_coord_type()

        if self.dim != 2 and self.grid:
            print('WARNING: grid == True is not allowed for dim != 2 (yet).'
                  'Turning grid = False.')
            self.grid = False

    def _verify_coord_type(self, coord_type):
        if isinstance(coord_type, str):
            coord_type = (coord_type,)

        if self.dim == 0:
            ret_coord_type = ()
        elif self.dim == 1:
            if coord_type is None:
                ret_coord_type = ('psi_n',)
            elif tuple(coord_type) in self._valid_coordinates_1d:
                ret_coord_type = tuple(coord_type)
            else:
                ret_coord_type = ('psi_n',)
                print("WARNING: _coord_type_input is not correct. \n"
                      "{} is not allowed \n"
                      "Force set _coord_type_input = ('psi_n',)"
                      .format(tuple(coord_type)))
        elif self.dim == 2:
            if coord_type is None:
                ret_coord_type = ('R', 'Z')
            elif tuple(coord_type) in self._valid_coordinates_2d:
                ret_coord_type = tuple(coord_type)
            elif tuple(coord_type[::-1]) in self._valid_coordinates_2d:
                ret_coord_type = tuple(coord_type[::-1])
            else:
                ret_coord_type = ('R', 'Z')
                print("WARNING: _coord_type_input is not correct. \n"
                      "{} is not allowed \n"
                      "Force set _coord_type_input = ('R', 'Z')"
                      .format(tuple(coord_type)))
        elif self.dim == 3:
            if coord_type is None:
                ret_coord_type = ('R', 'Z', 'phi')
            elif tuple(coord_type) in self._valid_coordinates_3d:
                ret_coord_type = tuple(coord_type)
            else:
                ret_coord_type = ('R', 'Z', 'phi')
                print("WARNING: _coord_type_input is not correct. \n"
                      "{} is not allowed \n"
                      "Force set _coord_type_input = ('R', 'Z', 'phi')"
                      .format(tuple(coord_type)))

        else:
            raise ValueError('Operation in {} space has not be en allowed yet. Sorry.'
                             .format(self.dim))
        # todo: make order dependent!
        return ret_coord_type

    def _incompatible_dimension_error(self, dim):
        raise ValueError('Operation in {} space has not be en allowed yet. Sorry.'
                         .format(dim))

    def _convert_to_default_coord_type(self):
        if self.dim == 0:
            return
        elif self.dim == 1:
            # convert to psi_n
            if self._coord_type_input == ('psi_n',):
                self.x1 = self._x1_input
            elif self._coord_type_input == ('psi',):
                psi = self._x1_input
                self.x1 = (psi - self._eq._psi_axis) / \
                          (self._eq._psi_lcfs - self._eq._psi_axis)

            elif self._coord_type_input == ('rho',):
                self.x1 = self._x1_input ** 2
            else:
                raise ValueError('This should not happen.')
        elif self.dim == 2:
            # only (R, Z) coordinates are implemented now
            if self._coord_type_input == ('R', 'Z'):
                self.x1 = self._x1_input
                self.x2 = self._x2_input
            elif self._coord_type_input == ('r', 'theta'):
                # todo: COCOS
                r_mgax, z_mgax = self._eq._mg_axis
                self.x1 = r_mgax + self._x1_input * np.cos(self._x2_input)
                self.x2 = z_mgax + self._x1_input * np.sin(self._x2_input)
        elif self.dim == 3:
            # only (R, Z) coordinates are implemented now
            if self._coord_type_input == ('R', 'Z', 'phi'):
                self.x1 = self._x1_input
                self.x2 = self._x2_input
                self.x3 = self._x3_input
            elif self._coord_type_input == ('X', 'Y', 'Z'):
                # todo: COCOS
                # R(1)**2 = X(1)**2 + Y(2)**2
                # Z(2) = Z(3)
                # phi(3) = atan2(Y(2), X(1)]
                self.x1 = np.sqrt(self._x1_input**2 + self._x2_input**2)
                self.x2 = self._x3_input
                self.x3 = np.arctan2(self._x2_input, self._x1_input)
