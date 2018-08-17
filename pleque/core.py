import numpy as np
import xarray


class FluxFuncs:
    # def interpolate(self, coords, data)
    # def interpolate(self, R, Z, data):
    #     pass

    def __init__(self, equi):
        # _flux_funcs = ['psi', 'rho']
        _flux_funcs = ['psi_n', 'psi', 'rho']
        self._equi = equi
        # self.__dict__.update(_flux_funcs)  # at class level?
        for fn in _flux_funcs:
            setattr(self, fn, getattr(self._equi, fn))  # methods are bound to _equi

    def add_flux_func(self, name, data, *coordinates, R=None, Z=None, psi_n=None, coord_type=None,
                      **coords):
        from scipy.interpolate import UnivariateSpline
        if R is not None and Z is not None:
            psi_n = self.psi_n(R=R, Z=Z)
        # interp = interpolate(psi_n, data)
        interp = UnivariateSpline(psi_n, data, s=0, k=3)
        setattr(self, '_interp_' + name, interp)

        def new_func(self: Equilibrium, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, **coords):
            if R is not None and Z is not None:
                psi_n = self.psi_n(R=R, Z=Z)
            return interp(psi_n)

        setattr(type(self), name, new_func)


class Equilibrium(object):
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
                 spline_order=5,
                 spline_smooth=0,
                 cocos=-1,
                 verbose=True
                 ):
        """

        Optional argumets may help the initialization.
    
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
        self._spline_order = 5
        self._cocos = cocos

        # todo: resolve this from input
        self._Bpol_sign = 1

        r = basedata.R.data
        z = basedata.Z.data
        psi = basedata.psi.transpose('R', 'Z').data

        self.r_min = np.min(r)
        self.r_max = np.max(r)
        self.z_min = np.min(z)
        self.z_max = np.max(z)

        if verbose:
            print('--- Generate 2D spline ---')

        # generate spline:
        # todo: first assume r, z are ascending:
        spl = RectBivariateSpline(r, z, psi, kx=spline_order, ky=spline_order,
                                  s=spline_smooth)
        self._spl_psi = spl

        # find extremes:
        self.__find_extremes__()

        # generate 1d profiles:
        if (self._psi_lcfs - self._psi_axis > 0):
            self._psi_sign = +1
        else:
            self._psi_sign = -1

        psi_n = basedata.psi_n.data
        pressure = basedata.pressure.data
        fpol = basedata.fpol.data
        self.BvacR = fpol[-1]

        self._fpol_spl = UnivariateSpline(psi_n, fpol, k=3, s=1)
        self._df_dpsin_spl = self._fpol_spl.derivative()
        self._pressure_spl = UnivariateSpline(psi_n, pressure, k=3, s=1)
        self._dp_dpsin_spl = self._pressure_spl.derivative()

        self.fluxfuncs.add_flux_func('fpol', fpol, psi_n=psi_n)
        self.fluxfuncs.add_flux_func('pressure', pressure, psi_n=psi_n)

    def psi(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        """
        Psi value

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        coord = Coordinates(self, *coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return coord.psi

    def psi_n(self, *coordinates, R=None, Z=None, psi=None, coord_type=None, grid=True, **coords):
        coord = Coordinates(self, *coordinates, R=R, Z=Z, psi=psi, coord_type=coord_type, grid=grid, **coords)
        return coord.psi_n

    @property
    def _diff_psi_n(self):
        return 1 / (self._psi_lcfs - self._psi_axis)

    def rho(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = Coordinates(self, *coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return np.sqrt(coord.psi_n)

    def pressure(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = Coordinates(self, *coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._pressure_spl(coord.psi_n)

    def pprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = Coordinates(self, *coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._dp_dpsin_spl(coord.psi_n) * self._diff_psi_n

    def fpol(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = Coordinates(self, *coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        # todo use in_plasma
        mask_out = coord.psi_n > 1
        fpol = self._fpol_spl(coord.psi_n)
        fpol[mask_out] = self.BvacR
        return fpol

    def fprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = Coordinates(self, *coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        mask_out = coord.psi_n > 1
        fprime = self._df_dpsin_spl(coord.psi_n) * self._diff_psi_n
        fprime[mask_out] = 0
        return fprime

    def ffprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = Coordinates(self, *coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
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

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return: Absolute value of magnetic field in Tesla.
        """
        coord = Coordinates(self, *coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        B_R = self.B_R(R=coord.R, Z=coord.Z)
        B_Z = self.B_Z(R=coord.R, Z=coord.Z)
        B_T = self.B_tor(R=coord.R, Z=coord.Z)
        B_abs = np.sqrt(B_R ** 2 + B_Z ** 2 + B_T ** 2)

        return B_abs

    def calc_gridcoords(self, rbase, ybase, dim = "size"):
        """
        Function which returns 2d grid with requested step/dimensions generated over the reconstruction space.
        :param rbase: float or int,  size of step or grid dimension size depending on dim
        :param zbase: float or int, size of step or grid dimension size depending on dim
        :param dim: list or string, If "step" then rbase, ybase are interpreted as requested step size in the grid.
        If "size" then rbase, zbase interpreted as requested grid sizes.
        :return:
        """

        if isinstance(dim,(list, tuple, np.ndarray)) and len(dim) == 2:
            if dim[0] == "step":
                r = np.arange(self._basedata.R.min(),self._basedata.R.max(), rbase)
            elif dim[0] == "size":
                r = np.linspace(self._basedata.R.min(),self._basedata.R.max(), ybase)
            else:
                raise ValueError("Wrong dim[0] value passed")

            if dim[1] == "step":
                z = np.arange(self._basedata.Z.min(),self._basedata.R.max(), rbase)
            elif dim[1] == "dim":
                z = np.linspace(self._basedata.Z.min(),self._basedata.Z.max(), ybase)
            else:
                raise ValueError("Wrong dim[1] value passed")
        elif isinstance(dim,str):
            if dim == "step":
                r = np.arange(self._basedata.R.min(),self._basedata.R.max(), rbase)
                z = np.arange(self._basedata.Z.min(),self._basedata.R.max(), rbase)
            elif dim == "size":
                r = np.linspace(self._basedata.R.min(),self._basedata.R.max(), ybase)
                z = np.linspace(self._basedata.Z.min(),self._basedata.Z.max(), ybase)
            else:
                raise ValueError("Wrong dim value passed")
        else:
            raise ValueError("Wrong dim value passed")

        return r, z

    # todo: resolve the grids
    def B_R(self, *coordinates, R=None, Z=None, coord_type=('R', 'Z'), grid=True, **coords):
        """
        Poloidal value of magnetic field in Tesla.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        coord = Coordinates(self, *coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        if grid:
            Rs, Zs = np.meshgrid(R, Z)
            Rs = Rs.T
        else:
            Rs = R
        return -self._spl_psi(R, Z, dy=1, grid=grid) / Rs * self._Bpol_sign

    def B_Z(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Poloidal value of magnetic field in Tesla.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        coord = Coordinates(self, *coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        if grid:
            Rs, Zs = np.meshgrid(R, Z)
            Rs = Rs.T
        else:
            Rs = R

        return self._spl_psi(R, Z, dx=1, grid=grid) / Rs * self._Bpol_sign

    def B_pol(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Absolute value of magnetic field in Tesla.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        coord = Coordinates(self, *coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        B_R = self.B_R(R=R, Z=Z)
        B_Z = self.B_Z(R=R, Z=Z)
        B_pol = np.sqrt(B_R ** 2 + B_Z ** 2)
        return B_pol

    def B_tor(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        """
        Toroidal value of magnetic field in Tesla.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        coord = Coordinates(self, *coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        if grid:
            R_mesh = R[:, None]
        else:
            R_mesh = R
        return self.fpol(R=R, Z=Z, grid=grid) / R_mesh

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

    def in_first_wall(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        from pleque.utils.surfaces import point_in_first_wall
        if grid:
            r_mesh, z_mesh = np.meshgrid(R, Z)
            points = np.vstack((r_mesh.ravel(), z_mesh.ravel())).T
        else:
            points = np.vstack((R, Z)).T

        mask_in = point_in_first_wall(self, points)
        return mask_in

    def in_lcfs(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        from pleque.utils.surfaces import point_inside_curve
        if grid:
            r_mesh, z_mesh = np.meshgrid(R, Z)
            points = np.vstack((r_mesh.ravel(), z_mesh.ravel())).T
        else:
            points = np.vstack((R, Z)).T

        mask_in = point_inside_curve(points, self._lcfs)
        return mask_in

    def __find_extremes__(self):
        from scipy.signal import argrelmin
        from scipy.optimize import minimize


        # for sure not the best algorithm ever...
        rs = np.linspace(self.r_min, self.r_max, 120)
        zs = np.linspace(self.z_min, self.z_max, 130)

        psi = self._spl_psi(rs, zs)
        psi_x = self._spl_psi(rs, zs, dx=1, dy=0)
        psi_y = self._spl_psi(rs, zs, dx=0, dy=1)
        psi_xysq = psi_x ** 2 + psi_y ** 2
        psi_xy = self._spl_psi(rs, zs, dx=1, dy=1)

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
                    bounds = ((np.max((self.r_min, r_ex - 0.1)),
                               np.min((self.r_max, r_ex + 0.1))),
                              (np.max((self.z_min, z_ex - 0.1)),
                               np.min((self.z_max, z_ex + 0.1))))

                    res = minimize(psi_xysq_func, x0, bounds=bounds)
                    r_ex2 = res['x'][0]
                    z_ex2 = res['x'][1]

                    #                    psi_xyabs = np.abs(psi_xy[ar, az])
                    psi_xyopt = np.abs(self._spl_psi(r_ex2, z_ex2, dx=1, dy=1, grid=False)) ** 2

                    if psi_xyopt < 0.1:
                        # plt.plot(rs[ar], zs[az], 'o', markersize=10, color='b')
                        # plt.plot(r_ex2, z_ex2, 'o', markersize=8, color='C4')
                        o_points.append((r_ex2, z_ex2))
                    else:
                        # plt.plot(rs[ar], zs[az], 'x', markersize=10, color='r')
                        # plt.plot(r_ex2, z_ex2, 'x', markersize=8, color='C5')
                        x_points.append((r_ex2, z_ex2))

        # todo: After beeing function written, check whether are points inside limiter

        # First identify the o-point nearest the operation range as center of plasma
        r_centr = (self.r_min + self.r_max) / 2
        z_centr = (self.z_min + self.z_max) / 2
        o_points = np.array(o_points)
        x_points = np.array(x_points)

        op_dist = (o_points[:, 0] - r_centr) ** 2 + (o_points[:, 1] - z_centr) ** 2
        # assume that psi value has its minimum in the center
        op_psiscale = self._spl_psi(o_points[:, 0], o_points[:, 1], grid=False)
        op_psiscale = 1 + (op_psiscale - np.min(op_psiscale)) / (np.max(op_psiscale) - np.min(op_psiscale))

        sortidx = np.argsort(op_dist * op_psiscale)
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
                print('>>> found lower x-point configuration')
            v = v[v[:, 1] > self._x_point[1], :]
            v = v[v[:, 1] < self._x_point2[1], :]

        else:
            if self._verbose:
                print('>>> found upper x-point configuration')
            v = v[v[:, 1] < self._x_point[1], :]
            v = v[v[:, 1] > self._x_point2[1], :]

        mask_in = self.in_first_wall(R=v[:, 0], Z=v[:, 1], grid=False)
        v = v[mask_in, :]

        self._lcfs = v

    @property
    def fluxfuncs(self):
        return FluxFuncs(self)  # filters out methods from self


class FluxSurface:
    def __init__(self, ):
        pass


class Coordinates(object):

    def __init__(self, equilibrium: Equilibrium, *coordinates, coord_type=None, grid=False, **coords):
        self._eq = equilibrium
        self._valid_coordinates = {'R', 'Z', 'psi_n', 'psi', 'rho'}
        self._valid_coordinates_1d = {('psi_n',), ('psi',), ('rho',)}
        self._valid_coordinates_2d = {('R', 'Z')}
        self.dim = -1  # init only
        self.grid = grid

        self.__evaluate_input__(*coordinates, coord_type=coord_type, **coords)

    def __call__(self, *args, **kwargs):
        pass

    def __iter__(self):
        pass

    def sort(self, order):
        pass

    @property
    def R(self):
        if self.dim == 2:
            return self.x1

    @property
    def Z(self):
        if self.dim == 2:
            # todo
            return self.x2

    @property
    def psi(self):
        if self.dim == 1:
            return self._eq._psi_axis + self.x1 * (self._eq._psi_lcfs - self._eq._psi_axis)
        elif self.dim == 2:
            return self._eq._spl_psi(self.x1, self.x2, grid=self.grid)

    @property
    def psi_n(self):
        if self.dim == 1:
            return self.x1
        elif self.dim == 2:
            return (self.psi - self._eq._psi_axis) / (self._eq._psi_lcfs - self._eq._psi_axis)

    @property
    def rho(self):
        return np.sqrt(self.psi_n)

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

    def getAs(self, coord_type=None):
        if self.dim == 0:
            return np.array(())
        coord_type_ = self._verify_coord_type(coord_type)

        if self.dim < len(coord_type):
            raise ValueError('Can not return 2d from 1d data.')
        raise ValueError('Not implemented yet.')

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

                if isinstance(xy, Coordinates):
                    # todo: ask Ondra (!)
                    print('ERROR: not implemented yet')
                    raise ValueError('not implemented yet')
                    pass
                elif isinstance(xy, Iterable):
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
                    else:
                        self._incompatible_dimension_error(self.dim)
                else:
                    # 1d, one number
                    self.dim = 1
                    self._x1_input = np.array([xy])
                pass
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

            else:
                self._incompatible_dimension_error(len(coordinates))

            self._coord_type_input = self._verify_coord_type(coord_type)
        self._convert_to_default_coord_type()

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
            self.x1 = self._x1_input
            self.x2 = self._x2_input
