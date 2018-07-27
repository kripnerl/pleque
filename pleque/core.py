import numpy as np
import xarray


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
        from scipy.interpolate import RectBivariateSpline

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

        basedata = basedata.transpose('R', 'Z')

        r = basedata.R.data
        z = basedata.Z.data
        psi = basedata.psi.data

        self.r_min = np.min(r)
        self.r_max = np.max(r)
        self.z_min = np.min(z)
        self.z_max = np.max(z)

        if verbose:
            print('Generate 2D spline')

        # generate spline:
        # todo: first assume r, z are ascending:
        spl = RectBivariateSpline(r, z, psi, kx=spline_order, ky=spline_order, s=0)
        self._spl = spl

        # find extremes:
        self._find_extremes_

    def psi(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
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
        return self._spl(R, Z, grid=grid)

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

        pass

    def B_R(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Poloidal value of magnetic field in Tesla.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        if grid:
            Rs, Zs = np.meshgrid(R, Z)
            Rs = Rs.T
        else:
            Rs = R
        return -self._spl(R, Z, dy=1, grid=grid) / Rs * self._Bpol_sign

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
        if grid:
            Rs, Zs = np.meshgrid(R, Z)
            Rs = Rs.T
        else:
            Rs = R

        return self._spl(R, Z, dx=1, grid=grid) / Rs * self._Bpol_sign

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
        pass

    def B_tor(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Toroidal value of magnetic field in Tesla.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param coords:
        :return:
        """
        B_R = self.B_R(R=R, Z=Z)
        B_Z = self.B_Z(R=R, Z=Z)
        B_pol = np.sqrt(B_R ** 2 + B_Z ** 2)
        return B_pol

    @property
    def _find_extremes_(self):
        from scipy.signal import argrelmin
        from scipy.optimize import minimize

        # for sure not the best algorithm ever...
        rs = np.linspace(self.r_min, self.r_max, 200)
        zs = np.linspace(self.z_min, self.z_max, 220)

        psi = self.psi(R=rs, Z=zs)
        psi_x = self._spl(rs, zs, dx=1, dy=0)
        psi_y = self._spl(rs, zs, dx=0, dy=1)
        psi_xysq = psi_x ** 2 + psi_y ** 2
        psi_xy = self._spl(rs, zs, dx=1, dy=1)

        mins0 = argrelmin(psi_xysq, axis=0)  # type: Tuple[int]
        mins1 = argrelmin(psi_xysq, axis=1)  # type: Tuple[int]

        import matplotlib.pyplot as plt
        # plt.figure()
        # plt.pcolormesh(rs, zs, psi.T)
        # plt.plot(mins0[0], mins0[1])
        # plt.contour(rs, zs, psi_xy.T, [0], colors='C4', ls='--')
        # plt.show()

        def psi_xysq_func(x):
            return self._spl(x[0], x[1], dx=1, dy=0, grid=False) ** 2 \
                   + self._spl(x[0], x[1], dx=0, dy=1, grid=False) ** 2

        x_points = []
        o_points = []

        for i, (ar, az) in enumerate(zip(mins0[0], mins0[1])):
            for j, (br, bz) in enumerate(zip(mins1[0], mins1[1])):
                if ar == br and az == bz:
                    r_ex = rs[ar]
                    z_ex = zs[az]
                    psi_xyabs = np.abs(psi_xy[ar, az])

                    # minimize in the vicinity:
                    bounds = ((np.max((self.r_min, r_ex - 0.1)),
                               np.min((self.r_max, r_ex + 0.1))),
                              (np.max((self.z_min, z_ex - 0.1)),
                               np.min((self.z_max, z_ex + 0.1))))

                    res = minimize(psi_xysq_func, (r_ex, z_ex), bounds=bounds)
                    r_ex2 = res['x'][0]
                    z_ex2 = res['x'][1]
                    psi_xyopt = np.abs(self._spl(r_ex2, z_ex2, dx=1, dy=1, grid=False))
                    print()
                    print(res)

                    if psi_xyopt < 0.1:
                        # plt.plot(rs[ar], zs[az], 'o', markersize=10, color='b')
                        # plt.plot(r_ex2, z_ex2, 'o', markersize=8, color='C4')
                        o_points.append((r_ex2, z_ex2))
                    else:
                        # plt.plot(rs[ar], zs[az], 'x', markersize=10, color='r')
                        # plt.plot(r_ex2, z_ex2, 'x', markersize=8, color='C5')
                        x_points.append((r_ex2, z_ex2))

        # Identify the o-point nearest the operation range as center of plasma
        r_centr = (self.r_min + self.r_max) / 2
        z_centr = (self.z_min + self.z_max) / 2
        o_points = np.array(o_points)
        x_points = np.array(x_points)

        op_dist = (o_points[:, 0] - r_centr) ** 2 + (o_points[:, 1] - z_centr) ** 2
        idx = np.argmin(op_dist)
        self._mg_axis = o_points[idx]
        self._psi_axis = np.asscalar(self.psi(R=self._mg_axis[0], Z=self._mg_axis[1]))

        # identify THE x-point as the x-point nearest in psi value to mg_axis
        # todo: Ensure that the psi function between x-point and o-point is monotonic (!)

        psi_diff = np.zeros(x_points.shape[0])
        for i in np.arange(x_points.shape[0]):
            rxp = x_points[i, 0]
            zxp = x_points[i, 1]
            psi_xp = np.asscalar(self.psi(R=rxp, Z=zxp, grid=False))
            psi_diff[i] = np.abs(psi_xp - self._psi_axis)

        # idx = np.argmin(psi_diff)
        sortidx = np.argsort(psi_diff)

        self._x_point = x_points[sortidx[0]]
        self._psi_lcfs = np.asscalar(self.psi(R=self._x_point[0], Z=self._x_point[1]))

        self._x_point2 = x_points[sortidx[1]]
        self._psi_xp2 = np.asscalar(self.psi(R=self._x_point2[0], Z=self._x_point2[1]))

        # get lcfs, for now using matplotlib contour line

        plt.figure(1111)
        cl = plt.contour(rs, zs, psi.T, [self._psi_lcfs])
        paths = cl.collections[0].get_paths()
        v = np.concatenate([p.vertices for p in paths], axis=0)
        plt.close(1111)

        if self._x_point[1] < self._x_point2[1]:
            if self._verbose:
                print('lower x-point configuration')
            v = v[v[:, 1] > self._x_point[1], :]
            v = v[v[:, 1] < self._x_point2[1], :]

        else:
            if self._verbose:
                print('upper x-point configuration')
            v = v[v[:, 1] < self._x_point[1], :]
            v = v[v[:, 1] > self._x_point2[1], :]

        self._lcfs = v
