from collections.abc import Sequence

import numpy as np
import xarray

from pleque.utils.decorators import deprecated

from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from pleque.core import Coordinates
from pleque.utils.tools import arglis
from pleque.core import FluxFunctions, Surface  # , FluxSurface
from pleque.core import cocos as cc
import pleque.utils.equi_tools as eq_tools
import pleque.utils.surfaces as surf


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
    #              cocos=3: int,
    #             ):
    def __init__(self,
                 basedata: xarray.Dataset,
                 first_wall=None,
                 mg_axis=None,
                 psi_lcfs=None,
                 x_points=None,
                 strike_points=None,
                 init_method="hints",
                 spline_order=3,
                 spline_smooth=0,
                 cocos=3,
                 verbose=True
                 ):
        """
        Equilibrium class instance should be obtained generally by functions in pleque.io
        package.

        Optional arguments may help the initialization.

        :param basedata: xarray.Dataset with psi(R, Z) on a rectangular R, Z grid, f(psi_norm), p(psi_norm)
                         f = B_tor * R
        :param first_wall: array-like (Nwall, 2)  required for initialization in case of limiter configuration.
        :param mg_axis: suspected position of the o-point
        :param psi_lcfs:
        :param x_points:
        :param strike_points:
        :param init_method: str On of ("full", "hints", "fast_forward").
                            If "full" no hints are taken and module tries to recognize all critical points itself.
                            If "hints" module use given optional arguments as a help with initialization.
                            If "fast-forward" module use given optional arguments as final and doesn't try to correct.
                            *Note:* Only "hints" method is currently tested.
        :param spline_order:
        :param spline_smooth:
        :param cocos: At the moment module assume cocos to be 3 (no other option). The implemetnation is not fully
                      working. Be aware of signs in the module!
        :param verbose:
        """

        if verbose:
            print('---------------------------------')
            print('Equilibrium module initialization')
            print('---------------------------------')

        self._basedata = basedata
        self._verbose = verbose
        self._mg_axis = mg_axis
        self._psi_lcfs = psi_lcfs
        self._x_points = x_points
        self._strike_points = strike_points
        self._spline_order = spline_order
        # TODO TODO TODO
        self._init_method = init_method
        self._cocos = cocos
        self._cocosdic = cc.cocos_coefs(cocos)

        # todo: resolve this from input (for COCOS time) TODO TODO TODO
        self._Bpol_sign = 1

        r = basedata.R.values
        z = basedata.Z.values
        psi = basedata.psi.transpose('R', 'Z').values

        if first_wall is None:
            if 'first_wall' in basedata:
                self._first_wall = basedata["first_wall"]
            elif 'R_first_wall' in basedata and 'Z_first_wall' in basedata:
                self._first_wall = np.array([basedata.R_first_wall.values, basedata.Z_first_wall.values]).T
            else:
                rwall_min = np.min(r)
                rwall_max = np.max(r)
                zwall_min = np.min(z)
                zwall_max = np.max(z)

                dr = rwall_max - rwall_min
                dz = zwall_max - zwall_min

                # todo: remove this if possible
                # lets reduce the wall a bit to be have some plasma behind the wall
                rwall_min += dr / 100
                rwall_max -= dr / 100
                zwall_min += dz / 100
                zwall_max -= dz / 100

                corners = np.array(
                    [[rwall_min, zwall_max], [rwall_max, zwall_max], [rwall_max, zwall_min], [rwall_min, zwall_min]])
                newwall_r = []
                newwall_z = []
                for i in range(-1, 3):
                    rs = np.linspace(corners[i, 0], corners[i + 1, 0], 20)
                    zs = np.linspace(corners[i, 1], corners[i + 1, 1], 20)
                    newwall_r += list(rs)
                    newwall_z += list(zs)
                self._first_wall = np.stack((newwall_r, newwall_z)).T
        else:
            self._first_wall = first_wall

        if 'time' in basedata:
            self.time = basedata['time'].values
        else:
            self.time = -1

        if 'time_unit' in basedata:
            self.time_unit = basedata['time_unit']
        else:
            self.time_unit = "ms"

        if 'shot' in basedata:
            # Shot number will be strictly integer
            self.shot = int(basedata['shot'])
        else:
            self.shot = 0

        # todo: other machine-related informations

        self.R_min = np.min(r)
        self.R_max = np.max(r)
        self.Z_min = np.min(z)
        self.Z_max = np.max(z)

        # TODO: allow FFprime, ffprime, pprime and other on the input
        psi_n = basedata.psi_n.values

        pressure = None
        pprime = None

        if 'pprime' in basedata:
            pprime = basedata.pprime.values
        if 'pressure' in basedata:
            pressure = basedata.pressure.values

        self.F0 = None
        # Try to find F0 in basedata:
        if 'F0' in basedata:
            self.F0 = basedata['F0']
            if isinstance(self.F0, xarray.DataArray):
                self.F0 = np.asscalar(self.F0.values)
        elif 'F0' in basedata.attrs:
            self.F0 = basedata.attrs['F0']

        F = None
        FFprime = None

        if 'FFprime' in basedata:
            FFprime = basedata.FFprime.values
        if 'F' in basedata:
            F = basedata.F.values

        # Other attempts to identify F0:
        if self.F0 is None:
            if F is not None:
                self.F0 = F[-1]

            elif 'B0' in basedata and 'R0' in basedata:
                self.F0 = basedata['B0'] * basedata['R0']
            elif 'B0' in basedata.attrs and 'R0' in basedata.attrs:
                self.F0 = basedata.attrs['B0'] * basedata.attrs['R0']


        # ---------------------------
        # --- Generate psi spline ---
        # ---------------------------
        if verbose:
            print('--- Generate 2D spline ---')

        spl = RectBivariateSpline(r, z, psi, kx=spline_order, ky=spline_order,
                                  s=spline_smooth)
        self._spl_psi = spl

        # -------------------------------
        # ---- Find critical points -----
        # -------------------------------
        if verbose:
            print('--- Looking for critical points ---')

        rs = np.linspace(self.R_min, self.R_max, 300)
        zs = np.linspace(self.Z_min, self.Z_max, 400)

        x_points, o_points = eq_tools.find_extremes(rs, zs, self._spl_psi)

        r_lim = (self.R_min, self.R_max)
        z_lim = (self.Z_min, self.Z_max)

        self._mg_axis, sortidx = eq_tools.recognize_mg_axis(o_points, self._spl_psi, r_lim, z_lim, self._mg_axis)
        self._psi_axis = np.asscalar(self._spl_psi(self._mg_axis[0], self._mg_axis[1], grid=False))
        self._o_points = o_points[sortidx]
        self._o_points[0] = self._mg_axis

        # ------------------------------------------
        # Recognize x-point plasma vs limiter plasma
        # ------------------------------------------
        if verbose:
            print('--- Recognizing equilibrium type ---')

        # todo: use these two x-points in the future
        (xp1, xp2), sortidx = eq_tools.recognize_x_points(x_points, self._mg_axis, self._psi_axis, self._spl_psi,
                                                          r_lim, z_lim, self._psi_lcfs, self._x_points)

        self._x_point = xp1
        self._x_point2 = xp2

        if xp1 is None:
            self._psi_xp = None
        else:
            self._psi_xp = self._spl_psi(*xp1, grid=False)

        self._x_points = x_points[sortidx]
        if xp1 is not None:
            self._x_points[0] = xp1
        if xp2 is not None:
            self._x_points[1] = xp2

        limiter_plasma, limiter_point = eq_tools.recognize_plasma_type(self._x_point, self._first_wall,
                                                                       self._mg_axis, self._psi_axis, self._spl_psi)

        self._limiter_plasma = limiter_plasma
        self._limiter_point = limiter_point

        if self._verbose:
            if limiter_plasma:
                print(">> Limiter plasma found.")
            else:
                print(">> X-point plasma found.")

        self._psi_lcfs = self._spl_psi(*limiter_point, grid=False)

        # -----------------------
        # --- Plasma boundary ---
        # -----------------------

        rs = np.linspace(self.R_min, self.R_max, 700)
        zs = np.linspace(self.Z_min, self.Z_max, 1200)

        if limiter_plasma:
            self._strike_points = self._limiter_point[np.newaxis, :]
            self._contact_point = self._limiter_point
        else:
            self._contact_point = None
            if len(self._first_wall) < 4:
                self._strike_points = None
            else:
                self._strike_points = eq_tools.find_strike_points(self._spl_psi, rs, zs, self._psi_lcfs,
                                                                  self._first_wall)

        if self._verbose:
            print("--- Looking for LCFS: ---")

        close_lcfs = eq_tools.find_close_lcfs(self._psi_lcfs, rs, zs, self._spl_psi,
                                              self._mg_axis, self._psi_axis)

        while surf.fluxsurf_error(self._spl_psi, close_lcfs, self._psi_lcfs) > 1e-10:
            close_lcfs = eq_tools.find_surface_step(self._spl_psi, self._psi_lcfs, close_lcfs)

        if self._verbose:
            print("Relative LCFS error: {}".format(surf.fluxsurf_error(self._spl_psi, close_lcfs, self._psi_lcfs)))

        if not limiter_plasma:
            close_lcfs = surf.add_xpoint(xp1, close_lcfs, self._mg_axis)

        self._lcfs = close_lcfs

        # generate 1d profiles:
        if self._psi_lcfs - self._psi_axis > 0:
            self._psi_sign = +1
        else:
            self._psi_sign = -1

        Fprime = None
        if FFprime is not None:
            F = eq_tools.ffprime2f(FFprime, self._psi_axis, self._psi_lcfs, self.F0)
            Fprime = FFprime / F

        if pprime is not None:
            pressure = eq_tools.pprime2p(pprime, self._psi_axis, self._psi_lcfs)

        self.BvacR = self.F0

        # if p and F are not define, run vacuum-like discharge:
        self._vacuum = False
        if pressure is None or F is None:
            pressure = np.zeros_like(psi_n)
            F = np.zeros_like(psi_n)
            self._vacuum = True

        if verbose:
            print('--- Generate 1D splines ---')
        self._fpol_spl = UnivariateSpline(psi_n, F, k=3, s=0)

        if FFprime is None:
            self._df_dpsin_spl = self._fpol_spl.derivative()
            Fprime = self._df_dpsin_spl(psi_n) / (self._psi_lcfs - self._psi_axis)
            FFprime = F * Fprime

        self._pressure_spl = UnivariateSpline(psi_n, pressure, k=3, s=0)

        if pprime is None:
            self._dp_dpsin_spl = self._pressure_spl.derivative()
            pprime = self._dp_dpsin_spl(psi_n) / (self._psi_lcfs - self._psi_axis)

        self._pprime_spl = UnivariateSpline(psi_n, pprime, k=3, s=0)
        self._Fprime_spl = UnivariateSpline(psi_n, Fprime, k=3, s=0)

        self.fluxfuncs.add_flux_func('F', F, psi_n=psi_n)
        self.fluxfuncs.add_flux_func('FFprime', FFprime, psi_n=psi_n)

        self.fluxfuncs.add_flux_func('pressure', pressure, psi_n=psi_n)
        self.fluxfuncs.add_flux_func('pprime', pprime, psi_n=psi_n)

        if verbose:
            print('--- Mapping midplane to psi_n ---')
        self.__map_midplane2psi__()

        if verbose:
            print('--- Mapping pressure and f func to psi_n ---')

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

    def diff_psi(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=False, **coords):
        r"""
        Return the value of :math:`\nabla \psi`. It is positive/negative if the :math:`\psi` is increasing/decreasing.

        :param coordinates:
        :param R:
        :param Z:
        :param psi_n:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        ret = np.sqrt(self._spl_psi(coord.R, coord.Z, grid=coord.grid, dx=1) ** 2 +
                      self._spl_psi(coord.R, coord.Z, grid=coord.grid, dy=1) ** 2)
        ret *= np.sign(self._psi_lcfs - self._psi_axis)
        if coord.grid:
            ret = ret.T
        return ret

    def psi_n(self, *coordinates, R=None, Z=None, psi=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi=psi, coord_type=coord_type, grid=grid, **coords)
        return coord.psi_n

    @property
    def _diff_psi_n(self):
        """
        psi_2 - psi_1 = (psi_n_2 - psi_n_1)*1/_diff_psi_n
        :return: Scaling parameter between psi_n and psi
        """
        return 1 / (self._psi_lcfs - self._psi_axis)

    def rho(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return np.sqrt(coord.psi_n)

    def pressure(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._pressure_spl(coord.psi_n)

    def pprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._pprime_spl(coord.psi_n)

    def F(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        # todo use in_plasma
        mask_out = coord.psi_n > 1
        F = self._fpol_spl(coord.psi_n)
        F[mask_out] = self.BvacR
        return F

    def Fprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        '''

        :param coordinates:
        :param R:
        :param Z:
        :param psi_n:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        '''
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        mask_out = coord.psi_n > 1
        Fprime = self._Fprime_spl(coord.psi_n)
        Fprime[mask_out] = 0
        return Fprime

    def FFprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        mask_out = coord.psi_n > 1
        FFprime = self._fpol_spl(coord.psi_n) * self._Fprime_spl(coord.psi_n)
        FFprime[mask_out] = 0
        return FFprime

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

    # XXXXXX TODO TODO TODO
    @deprecated('The structure and behaviour of this function will change soon!\n'
                'to keep the same behaviour use `_flux_surface` instead.')
    def flux_surface(self, *coordinates, resolution=(1e-3, 1e-3), dim="step",
                     closed=True, inlcfs=True, R=None, Z=None, psi_n=None,
                     coord_type=None, **coords):
        return self._flux_surface(*coordinates, resolution=resolution, dim=dim,
                                  closed=closed, inlcfs=inlcfs, R=R, Z=Z, psi_n=psi_n,
                                  coord_type=coord_type, **coords)

    def _flux_surface(self, *coordinates, resolution=None, dim="step",
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
        :param resolution:  Iterable of size 2 or a number, default is [1e-3, 1e-3]. If a number is passed,
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

        coordinates = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, **coords)

        # get the grid for psi map to find the contour in.
        # todo: this is, at the moment, slowest part of the code

        grid = self.grid(resolution=resolution, dim=dim)

        # todo: to get lcfs, here is small trick. This should be handled better
        #       otherwise it may return crossed loop
        if np.isclose(coordinates.psi_n[0], 1) and inlcfs:
            psi_n = 1 - 1e-5
        else:
            psi_n = coordinates.psi_n[0]

        # create coordinates
        # coords = self.coordinates(R=R, Z=Z, grid=True, coord_type=["R", "Z"])

        # get the coordinates of the contours with requested leve and convert them into
        # instances of FluxSurface class

        contour = self._get_surface(grid, level=psi_n, norm=True)

        for i in range(len(contour)):
            contour[i] = self._as_fluxsurface(contour[i])

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

    def plot_geometry(self, axs=None, **kwargs):
        """
        Plots the the directions of angles, current and magnetic field.

        :axs = None or tuple of axes.
        :param kwargs: parameters passed to the `plot` routine.
        :return: tuple of axis (ax1, ax2)
        """
        import matplotlib.pyplot as plt

        if axs is None:
            fig, axs = plt.subplots(1, 2)

        fw = self.first_wall

        if len(fw) < 4:
            print('Warning: first wall is not sufficient. LCFS is used instead.')
            fw = self.lcfs

        R_min = np.min(fw.R)
        R_max = np.max(fw.R)
        R_mid = (R_min + R_max) / 2

        sig_ip = np.sign(self.I_plasma)
        sig_bt = np.sign(self.F0)

        #############
        # Top view: #
        #############

        ax1 = axs[0]
        ax1.set_title('Top view')
        phis = np.linspace(0, 2 * np.pi, endpoint=True)

        ax1.plot(R_min * np.cos(phis), R_min * (np.sin(phis)), 'k-')
        ax1.plot(R_max * np.cos(phis), R_max * (np.sin(phis)), 'k-')

        phi_dir = self.coordinates(R=R_mid, Z=0, phi=np.pi / 8)
        phi_dir_mg = self.coordinates(R=R_mid, Z=0, phi=np.pi + sig_bt * np.pi / 8)
        phi_dir_ip = self.coordinates(R=R_mid, Z=0, phi=3 * np.pi / 2 + sig_ip * np.pi / 8)

        ax1.arrow(R_mid, 0, phi_dir.X[0] - R_mid, phi_dir.Y[0],
                  width=0.005, length_includes_head=True, head_width=0.05)
        ax1.text(R_mid + R_mid / 14, 0, r'$\phi$',
                 ha='left', va='center')

        ax1.arrow(-R_mid, 0, phi_dir_mg.X[0] + R_mid, phi_dir_mg.Y[0],
                  width=0.005, length_includes_head=True, head_width=0.05)
        ax1.text(- (R_mid + R_mid / 14), 0, r'$B_\phi$',
                 ha='right', va='center')

        ax1.arrow(0, -R_mid, phi_dir_ip.X[0], phi_dir_ip.Y[0] + R_mid,
                  width=0.005, length_includes_head=True, head_width=0.05)
        ax1.text(0, - (R_mid + R_mid / 14), r'$j_\phi$',
                 ha='center', va='top')

        ax1.set_aspect('equal')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')

        ###########################
        # Poloidal cross section: #
        ###########################
        r0 = (R_max - self.magnetic_axis.R[0]) * 0.6
        pos0 = self.coordinates(r=r0, theta=0)
        theta0 = 0
        theta_dir = self.coordinates(r=r0, theta=theta0 + np.pi / 8)

        r1 = (self.magnetic_axis.R[0] - R_min) * 0.7
        r2 = (self.magnetic_axis.R[0] - R_min) * 0.45
        theta1 = theta2 = np.pi
        pos1 = self.coordinates(r=r1, theta=theta1)
        pos2 = self.coordinates(r=r2, theta=theta2)

        sig_theta = self._cocosdic['sigma_pol'] * self._cocosdic['sigma_cyl']
        sig_bpol = sig_theta * np.sign(self.B_Z(pos1))
        sig_jpol = sig_theta * np.sign(self.j_Z(pos2))

        theta_dir_bpol = self.coordinates(r=pos1.r[0], theta=pos1.theta[0] + sig_bpol * np.pi / 8)
        theta_dir_jpol = self.coordinates(r=pos2.r[0], theta=pos2.theta[0] + sig_jpol * np.pi / 8)


        ax2 = axs[1]
        ax2.set_title("Poloidal cross section")
        ax2.plot(fw.R, fw.Z, 'k-')
        if fw is not self.lcfs:
            ax2.plot(self.lcfs.R, self.lcfs.Z, 'C0--')
        ax2.plot(self.magnetic_axis.R, self.magnetic_axis.Z, 'C0o')

        ax2.arrow(pos0.R[0], pos0.Z[0], theta_dir.R[0] - pos0.R[0], theta_dir.Z[0] - pos0.Z[0],
                  width=0.005, head_width=0.03)
        ax2.text(pos0.R[0] + r0 / 14, pos0.Z[0], r'$\theta$',
                 ha='left', va='center')

        ax2.arrow(pos1.R[0], pos1.Z[0], theta_dir_bpol.R[0] - pos1.R[0], theta_dir_bpol.Z[0] - pos1.Z[0],
                  width=0.005, head_width=0.03)
        ax2.text(pos1.R[0] - r0 / 14, pos1.Z[0], r'$B_\theta$',
                 ha='right', va='center')

        ax2.arrow(pos2.R[0], pos2.Z[0], theta_dir_jpol.R[0] - pos2.R[0], theta_dir_jpol.Z[0] - pos2.Z[0],
                  width=0.005, head_width=0.03)
        ax2.text(pos2.R[0] + r0 / 14, pos2.Z[0], r'$j_\theta$',
                 ha='left', va='center')

        ax2.set_aspect('equal')

        ax2.set_xlabel('R [m]')
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Z [m]')

        return fig

    def plot_overview(self, ax=None, **kwargs):
        """
        Simple routine for plot of plasma overview
        :return:
        """
        return self._plot_overview(ax, **kwargs)

    def _plot_overview(self, ax=None, **kwargs):
        """
        Simple routine for plot of plasma overview
        :return:
        """
        from pleque.utils.plotting import plot_equilibrium
        # plt.figure()
        return plot_equilibrium(self, ax=ax, **kwargs)

    def grid(self, resolution=None, dim="step"):
        """
        Function which returns 2d grid with requested step/dimensions generated over the reconstruction space.

        :param resolution: Iterable of size 2 or a number. If a number is passed,
                           R and Z dimensions will have the same size or step (depending on dim parameter). Different R and Z
                           resolutions or dimension sizes can be required by passing an iterable of size 2.
                           If None, default grid of size (1000, 2000) is returned.
        :param dim: iterable of size 2 or string ('step', 'size'). Default is "step", determines the meaning
                    of the resolution.
                    If "step" used, values in resolution are interpreted as step length in psi poloidal map. If "size" is used,
                    values in resolution are interpreted as requested number of points in a dimension. If string is passed,
                    same value is used for R and Z dimension. Different interpretation of resolution for R, Z dimensions can be
                    achieved by passing an iterable of shape 2.
        :return: Instance of `Coordinates` class with grid data
        """
        if resolution is None:
            if not hasattr(self, '_default_grid'):
                # TODO THIS is slow now. Decrease resolution and then use find_fluxsurface_step (!!!)
                R = np.linspace(self._basedata.R.min(), self._basedata.R.max(), 1000)
                Z = np.linspace(self._basedata.Z.min(), self._basedata.Z.max(), 2000)
                self._default_grid = self.coordinates(R=R, Z=Z, grid=True)
            return self._default_grid
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
                    R = self._basedata.R.values
                elif dim[0] == "step":
                    R = np.arange(self._basedata.R.min(), self._basedata.R.max(), res_R)
                elif dim[0] == "size":
                    R = np.linspace(self._basedata.R.min(), self._basedata.R.max(), res_Z)
                else:
                    raise ValueError("Wrong dim[0] value passed")

                if res_Z is None:
                    Z = self._basedata.Z.values
                elif dim[1] == "step":
                    Z = np.arange(self._basedata.Z.min(), self._basedata.R.max(), res_R)
                elif dim[1] == "size":
                    Z = np.linspace(self._basedata.Z.min(), self._basedata.Z.max(), res_Z)
                else:
                    raise ValueError("Wrong dim[1] value passed")
            elif isinstance(dim, str):
                if dim == "step":
                    if res_R is None:
                        R = self._basedata.R.values
                    else:
                        R = np.arange(self._basedata.R.min(), self._basedata.R.max(), res_R)
                    if res_Z is None:
                        Z = self._basedata.Z.values
                    else:
                        Z = np.arange(self._basedata.Z.min(), self._basedata.Z.max(), res_Z)
                elif dim == "size":
                    if res_R is None:
                        R = self._basedata.R.values
                    else:
                        R = np.linspace(self._basedata.R.min(), self._basedata.R.max(), res_R)
                    if res_Z is None:
                        Z = self._basedata.Z.values
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
        cc_norm = self._cocosdic["sigma_cyl"] * self._cocosdic["sigma_Bp"] * 1 / (2 * np.pi) ** self._cocosdic["exp_Bp"]
        return cc_norm * self._spl_psi(coord.R, coord.Z, dy=1, grid=coord.grid).T / coord.R * self._Bpol_sign

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
        cc_norm = self._cocosdic["sigma_cyl"] * self._cocosdic["sigma_Bp"] * 1 / (2 * np.pi) ** self._cocosdic["exp_Bp"]
        return - cc_norm * self._spl_psi(coord.R, coord.Z, dx=1, grid=coord.grid).T / coord.R * self._Bpol_sign

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
        return self.F(coord) / coord.R

    def abs_q(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        """
        Absolute value of q.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        return np.abs(self.q(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords))

    def q(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        if not hasattr(self, '_q_spl'):
            self._init_q()
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return self._q_spl(coord.psi_n)

    def diff_q(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=False, **coords):
        """

        :param self:
        :param coordinates:
        :param R:
        :param Z:
        :param psi_n:
        :param coord_type:
        :param grid:
        :param coords:
        :return: Derivative of q with respect to psi.
        """
        if not hasattr(self, '_dq_dpsin_spl'):
            self._init_q()
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._dq_dpsin_spl(coord.psi_n) * self._diff_psi_n

    def tor_flux(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        """
        Calculate toroidal magnetic flux :math:`\Phi` from:

        .. math::
            q = \frac{\mathrm{d \Phi} }{\mathrm{d \psi}}

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """

        if not hasattr(self, '_q_anideriv_spl'):
            self._init_q()
        cc = self._cocosdic['sigma_Bp'] * self._cocosdic['sigma_pol'] / ((2 * np.pi) ** (1 - self._cocosdic['exp_Bp']))
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return cc * self._q_anideriv_spl(coord.psi_n) * (1 / self._diff_psi_n)

    def j_R(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        # todo test cocos here!
        # todo: test grid
        # todo: test test test
        from scipy.constants import mu_0

        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        cc = self._cocosdic['sigma_cyl']

        dpsi_dZ = self._spl_psi(coord.R, coord.Z, grid=coord.grid, dy=1)
        if coord.grid:
            dpsi_dZ = dpsi_dZ.T

        return - cc *self.Fprime(coord) / (coord.R * mu_0) * dpsi_dZ

    def j_Z(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        from scipy.constants import mu_0

        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        cc = self._cocosdic['sigma_cyl']

        dpsi_dR = self._spl_psi(coord.R, coord.Z, grid=coord.grid, dx=1)
        if coord.grid:
            dpsi_dR = dpsi_dR.T

        return cc * self.Fprime(coord) / (coord.R * mu_0) * dpsi_dR

    def j_pol(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        r"""
        Poloidal component of the current density.
        Calculated as

        .. math::
          \frac{f' \nabla \psi }{R \mu_0}

        [Wesson: Tokamaks, p. 105]

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        from scipy.constants import mu_0
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return self.Fprime(coord) / (coord.R * mu_0) * self.diff_psi(coord)

    def j_tor(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        r"""
        todo: to be tested

        Toroidal component of the current denisity.
        Calculated as

        .. math::
          R p' + \frac{1}{\mu_0 R} ff'

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        from scipy.constants import mu_0
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        cc_norm = - self._cocosdic["sigma_Bp"] * (2 * np.pi) ** self._cocosdic["exp_Bp"]
        return cc_norm * (coord.R * self.pprime(coord) + 1 / (mu_0 * coord.R) * self.FFprime(coord))

    def get_precise_lcfs(self):
        """
        Calculate plasma LCFS by field line tracing technique and save LCFS as
        instance property.

        :return:
        """
        from pleque.utils.surfaces import track_plasma_boundary

        lcfs = track_plasma_boundary(self, self._x_point)

        # todo debug this :) after this doesn't work eq.plot_overview()
        # self._lcfs = lcfs
        return lcfs

    @property
    def lcfs(self):
        if not hasattr(self, '_lcfs_fl'):
            if not surf.curve_is_closed(self._lcfs):
                self._lcfs = np.vstack((self._lcfs, self._lcfs[0]))
            self._lcfs_fl = self._as_fluxsurface(self._lcfs)

        return self._lcfs_fl

    @property
    def separatrix(self):
        """
        If the equilibrium is limited, returns lcfs. If it is diverted it returns separatrix flux surface

        :return:
        """
        if not self._limiter_plasma:
            if not hasattr(self, '_separatrix'):
                self._find_separatrix()
            return self._as_fluxsurface(self._separatrix)
        else:
            return self.lcfs

    def _find_separatrix(self):
        """
        Finds separatrix contour by finding contour with normalized poloidal flux going to 1 from right....
        The proceedure iterates 1+0.000001*counter and stops when the found contour has intersection points with the
        first wall contour.
        :return:
        """

        found = False
        cnt = 1
        # todo: This should be rewritten
        while not found and cnt < 101:
            psi_n = 1+1e-6*cnt
            cnt += 1
            separatrix = self._flux_surface(inlcfs=False, closed=False, psi_n=psi_n)
            selstrikepoints = []
            for j in separatrix:
                # todo: this is not separatrix... for example in limiter plasma
                intersection = np.array(self.first_wall._string.intersection(j._string))
                if len(intersection) > 0:
                    self._separatrix = j.as_array(("R", "Z"))
                    found = True

        return self._separatrix

    @property
    def contact_point(self):
        """
        Returns contact point as instance of coordinates for circular plasmas. Returns None otherwise.
        :return:
        """
        if self._contact_point is None:
            return None
        else:
            return self.coordinates(self._contact_point[0], self._contact_point[1])

    @property
    def strike_points(self):
        """
        Returns contact point if the equilibrium is limited. If the equilibrium is diverted it returns strike points.
        :return:
        """
        if self._strike_points is None:
            return None  # This should not happen if the wall consists of enough points
        else:
            return self.coordinates(self._strike_points[:, 0], self._strike_points[:, 1])

    @property
    def limiter_point(self):
        """
        The point which "limits" the LCFS of plasma. I.e. contact point in case of limiter plasma and x-point
        in case of x-point plasma.

        :return: Coordinates
        """
        return self.coordinates(self._limiter_point[0], self._limiter_point[1])

    @property
    def x_point(self):
        """
        Return x-point closest in psi to mg-axis if presented on grid. None otherwise.

        :return Coordinates
        """
        if self._x_point is None:
            return None
        else:
            return self.coordinates(*self._x_point)

    @property
    def first_wall(self):
        """
        If the first wall polygon is composed of 3 and more points Surface instance is returned.
        If the wall contour is composed of less than 3 points, coordinate instance is returned, because Surface can't
        be constructed
        :return:
        """
        if self._first_wall.shape[0] < 3:
            return self.coordinates(self._first_wall)
        else:
            first_wall = self._first_wall

            # first wall should be a closed contour
            # todo: this should be chacked in init
            if not surf.curve_is_closed(first_wall):
                first_wall = np.concatenate((first_wall, first_wall[0, :][None, :]), axis = 0)
            return Surface(self, first_wall)

    @property
    def magnetic_axis(self):
        return self.coordinates(self._mg_axis[0], self._mg_axis[1])

    @property
    def I_plasma(self):
        """
        Toroidal plasma current. Calculated as toroidal current through the LCFS.

        :return: (float) Value of toroidal plasma current.
        """
        if not hasattr(self, "_Ip"):
            self._Ip = self.lcfs.tor_current
        return self._Ip

    def coordinates(self, *coordinates, coord_type=None, grid=False, **coords):
        """
        Return instance of Coordinates. If instances of coordinates is already on the input, just pass it through.

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

    def _as_fluxsurface(self, *coordinates, coord_type=None, grid=False, **coords):
        """

        :param coordinates:
        :param coord_type:
        :param grid:
        :param coords:
        :return:
        """
        from pleque import FluxSurface
        if len(coordinates) >= 1 and isinstance(coordinates[0], FluxSurface):
            return coordinates[0]
        elif len(coordinates) >= 1 and isinstance(coordinates[0], Coordinates):
            coord = coordinates[0]
            return FluxSurface(self, coord.R, coord.Z)
        else:
            return FluxSurface(self, *coordinates, coord_type=coord_type, grid=grid, **coords)

    def in_first_wall(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        from pleque.utils.surfaces import points_inside_curve
        # if grid:
        #     r_mesh, z_mesh = np.meshgrid(R, Z)
        #     points = np.vstack((r_mesh.ravel(), z_mesh.ravel())).T
        # else:
        #     points = np.vstack((R, Z)).T
        # mask_in = point_in_first_wall(self, points)
        # return mask_in
        points = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)

        mask_in = points_inside_curve(points.as_array(), self._first_wall)
        if points.grid:
            mask_in = mask_in.reshape(len(points.x2), len(points.x1))
        return mask_in

    def in_lcfs(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        from pleque.utils.surfaces import points_inside_curve
        # if grid:
        #     r_mesh, z_mesh = np.meshgrid(R, Z)
        #     points = np.vstack((r_mesh.ravel(), z_mesh.ravel())).T
        # else:
        #     points = np.vstack((R, Z)).T
        points = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)

        mask_in = points_inside_curve(points.as_array(), self._lcfs)
        if points.grid:
            mask_in = mask_in.reshape(len(points.x2), len(points.x1))
        return mask_in

    def connection_length(self, *coordinates, R: np.array = None, Z: np.array = None,
                          coord_type=None, direction = 1, **coords):
        """
        Calculate connection length from given coordinates to first wall

        Todo: The field line is traced to min/max value of z of first wall, distance is calculated to the last
            point before first wall.
        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param direction: if positive trace field line in/cons the direction of magnetic field.
        :param stopper: (None, 'poloidal', 'z-stopper) force to use stopper. If None stopper is
                       automatically chosen based on psi_n coordinate.
        :param coords:
        :return:
        """
        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)
        traces = self.trace_field_line(coords, direction = direction)
        dists = []
        lines = []

        for t in traces:
            if t.psi_n[0] > 1:
                # todo: find first False!
                mask_in = self.in_first_wall(t)
                rzp = t.as_array()[mask_in, :]

                #todo: add intersection point!

                line_in = self.coordinates(rzp)
                dist = line_in.length

                dists.append(dist)
                lines.append(line_in)
            else:
                dists.append(np.infty)
                lines.append(None)

        return dists, lines

    def trace_field_line(self, *coordinates, R: np.array = None, Z: np.array = None,
                         coord_type=None, direction=1, stopper_method=None, in_first_wall=False, **coords):
        """
        Return traced field lines starting from the given set of at least 2d coordinates.
        One poloidal turn is calculated for field lines inside the separatrix. Outter field lines
        are limited by z planes given be outermost z coordinates of the first wall.

        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param direction: if positive trace field line in/cons the direction of magnetic field.
        :param stopper_method: (None, 'poloidal', 'z-stopper) force to use stopper. If None stopper is
                       automatically chosen based on psi_n coordinate.
        :param in_first_wall: if True the only inner part of field line is returned.
        :param coords:
        :return:

        """
        import pleque.utils.field_line_tracers as flt
        from scipy.integrate import solve_ivp

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)

        res = []

        # XXXNOW
        coords_rz = coords.as_array(dim=2)

        sigma_B0 = np.sign(self.F0)

        dphifunc = flt.dphi_tracer_factory(self.B_R, self.B_Z, self.B_tor)

        r_lims = [np.min(self.first_wall.R), np.max(self.first_wall.R)]
        z_lims = [np.min(self.first_wall.Z), np.max(self.first_wall.Z)]

        for i in np.arange(len(coords)):

            y0 = coords_rz[i]
            if coords.dim == 2:
                phi0 = 0
            else:
                phi0 = coords.phi[i]

            atol = 1e-6
            if self.is_xpoint_plasma:
                xp = self._x_point
                xp_dist = np.sqrt(np.sum((xp - y0) ** 2))
                atol = np.minimum(xp_dist * 1e-3, atol)

            if self._verbose:
                print('>>> tracing from: {:3f},{:3f},{:3f}'.format(y0[0], y0[1], phi0))
                print('>>> atol = {}'.format(atol))

            if stopper_method is None:
                if coords.psi_n[i] <= 1:
                    # todo: determine the direction (now -1) !!
                    if self._verbose:
                        print('>>> poloidal stopper is used')

                    # XXX Direction (TODO)
                    # XXX add these values to cocos dict!
                    # sign(dtheta/dphi) = sigma_pol * sign(I * B)
                    # dphidtheta = self._cocosdic['sigma_pol'] * np.sign(self.I_plasma) * np.sign(self.F0)
                    # print('dir: {}\nsigma_pol: {}\nsigma_tor: {}\nIp: {}\nF0: {}'.format(
                    #     direction, self._cocosdic['sigma_pol'], self._cocosdic['sigma_cyl'], self.I_plasma, self.F0
                    # ))
                    # print('------------------')

                    dphidtheta = np.sign(self.F0) * self._cocosdic['sigma_pol'] * self._cocosdic['sigma_cyl']
                    print('direction: {}'.format(direction))
                    print('dphidtheta: {}'.format(dphidtheta))

                    stopper_method = flt.poloidal_angle_stopper_factory(y0, self.magnetic_axis.as_array()[0],
                                                                        dphidtheta * direction)
                else:
                    if self._verbose:
                        print('>>> z-lim stopper is used')
                    stopper_method = flt.rz_coordinate_stopper_factory(r_lims, z_lims)
            elif stopper_method == 'z-stopper':
                if self._verbose:
                    print('>>> z-lim stopper is used')
                stopper_method = flt.rz_coordinate_stopper_factory(r_lims, z_lims)
            elif stopper_method == 'poloidal':
                if self._verbose:
                    print('>>> poloidal stopper is used')

                dphidtheta = np.sign(self.F0) * self._cocosdic['sigma_pol'] * self._cocosdic['sigma_cyl']
                stopper_method = flt.poloidal_angle_stopper_factory(y0, self.magnetic_axis.as_array()[0],
                                                                    dphidtheta * direction)

            # todo: define somehow sufficient tolerances
            sol = solve_ivp(dphifunc,
                            (phi0, direction * sigma_B0 * (2 * np.pi * 50 + phi0)),
                            y0,
                            #                            method='RK45',
                            method='LSODA',
                            events=stopper_method,
                            max_step=1e-2,  # we want high phi resolution
                            atol=atol,
                            rtol=1e-8,
                            )

            if self._verbose:
                print("{}, {}".format(sol.message, sol.nfev))

            phi = sol.t
            R, Z = sol.y

            fl = self.coordinates(R, Z, phi)

            # XXX add condirtion to stopper
            if in_first_wall:
                mask = self.in_first_wall(fl)

                imask = mask.astype(int)
                in_idxs = np.where(imask[:-1] - imask[1:] == 1)[0]

                last_idx = False
                if len(in_idxs) >= 1:
                    # Last point is still in (+1)
                    last_idx = in_idxs[0]
                    mask[last_idx + 1:] = False

                Rs = fl.R[mask]
                Zs = fl.Z[mask]
                phis = fl.phi[mask]

                intersec = self.first_wall.intersection(fl, dim=2)
                if intersec is not None and len(in_idxs) >= 1:
                    R_last = Rs[-1]
                    Z_last = Zs[-1]

                    inter_idx = np.argmin((intersec.R - R_last) ** 2 + (intersec.Z - Z_last) ** 2)

                    Rx = intersec.R[inter_idx]
                    Zx = intersec.Z[inter_idx]
                    # last_idx = len(phis) - 1

                    coef = np.sqrt((Rx - fl.R[last_idx]) ** 2 + (Zx - fl.Z[last_idx]) ** 2 /
                                   (fl.R[last_idx + 1] - fl.R[last_idx]) ** 2 +
                                   (fl.Z[last_idx + 1] - fl.Z[last_idx]) ** 2)

                    phix = fl.phi[last_idx] + coef * (fl.phi[last_idx + 1] - fl.phi[last_idx])

                    Rs = np.append(Rs, Rx)
                    Zs = np.append(Zs, Zx)
                    phis = np.append(phis, phix)

                fl = self.coordinates(Rs, Zs, phis)

            res.append(fl)

        return res


    def trace_flux_surface(self, *coordinates, s_resolution=1e-3, R=None,
                           Z=None, psi_n=None, coord_type=None, **coords):
        """
        Find a closed flux surface inside LCFS with requested values of psi or psi-normalized.


        TODO support open and/or flux surfaces outise LCFS, needs different stopper

        :param R:
        :param Z:
        :param psi_n:
        :param coord_type:
        :param coordinates: specifies flux surface to search for (by spatial point or values of psi or psi normalised).
                            If coordinates is spatial point (dim=2) then the trace starts at the midplane.
                            Coordinates.grid must be False.
        :param s_resolution: max_step in the distance along the flux surface contour
        :return: FluxSurface
        """
        import pleque.utils.field_line_tracers as flt
        from scipy.integrate import solve_ivp

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)
        if coords.dim == 1:
            coords = self.coordinates(R=self.magnetic_axis.R+coords.r_mid, Z=0)
        y0 = np.reshape([coords.R, coords.Z], (2,))

        ds_func = flt.ds_grad_psi_tracer_factory(self._spl_psi)
        # atol is square of s_resolution to offset the square distance
        stopper = flt.rz_target_s_min_stopper_factory(y0, coords.r_mid,
                                                      atol=s_resolution**2)

        sol = solve_ivp(ds_func,
                        (0, coords.r_mid*2*np.pi*4),
                        y0,
                        method='LSODA',
                        events=stopper,
                        max_step=s_resolution,
                        vectorized=True,
                        # rtol=1e-8,
                        )
        fs = self._as_fluxsurface(R=np.hstack([sol.y[0], sol.y[0,0]]),
                                  Z=np.hstack([sol.y[1], sol.y[1,0]]))
        fs._cum_length = np.hstack([sol.t, sol.t[-1]+s_resolution])  # TODO use some setter
        return fs


    @property
    def fluxfuncs(self):
        if not hasattr(self, '_fluxfunc'):
            self._fluxfunc = FluxFunctions(self)  # filters out methods from self
        return self._fluxfunc

    def to_geqdsk(self, file, nx=64, ny=128, q_positive=True):
        """
        Write a GEQDSK equilibrium file.

        :param file: str, file name
        :param nx: int
        :param ny: int
        """
        import pleque.io.geqdsk as geqdsk

        geqdsk.write(self, file, nx=nx, ny=ny, q_positive=q_positive)


    @property
    def cocos(self):
        """
        Number of internal COCOS representation.

        :return: int
        """
        return self._cocos

    @property
    def is_xpoint_plasma(self):
        """
        Return true for x-point plasma.

        :return: bool
        """
        return not self._limiter_plasma

    @property
    def is_limter_plasma(self):
        """
        Return true if the plasma is limited by point or some limiter point.

        :return: bool
        """
        return self._limiter_plasma

    def __map_midplane2psi__(self):
        from scipy.interpolate import UnivariateSpline

        r_mid = np.linspace(0, self.R_max - self._mg_axis[0], 100)
        psi_mid = self.psi(r_mid + self._mg_axis[0], self._mg_axis[1] * np.ones_like(r_mid), grid=False)

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

    def _init_q(self):
        psi_n = np.arange(0.01, 1, 0.005)
        qs = []

        if self._verbose:
            print("--- Generating q-splines ---")
        for i, pn in enumerate(psi_n):
            if self._verbose and np.mod(i, 20) == 0:
                print("{:.0f}%\r".format(pn / np.max(psi_n) * 100))
            surface = self._flux_surface(psi_n=pn)
            c = surface[0]
            qs.append(c.eval_q)
        qs = np.array(qs)

        self._q_spl = UnivariateSpline(psi_n, qs, s=0, k=3)
        self._dq_dpsin_spl = self._q_spl.derivative()
        self._q_anideriv_spl = self._q_spl.antiderivative()
