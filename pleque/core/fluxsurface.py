import numpy as np
from shapely import geometry

from pleque.core import Coordinates
from pleque.utils.decorators import *


class Surface(Coordinates):

    def __init__(self, equilibrium, *coordinates, coord_type=None, grid=False, **coords):
        """
        Calculates geometrical properties of a specified surface. To make the contour closed, the first and last points in
        the passed coordinates have to be the same.
        Instance is obtained by calling method `surface` in instance of `Equilibrium`.

        :param coords: Instance of coordinate class
        """

        super().__init__(equilibrium, *coordinates, coord_type=None, grid=False, **coords)

        points_RZ = self.as_array(('R', 'Z'))
        # closed surface has to have identical first and last points and then the shape is polygon
        # opened surface is linestring
        if np.isclose(points_RZ[0, 0], points_RZ[-1, 0]) and np.isclose(points_RZ[0, 1], points_RZ[-1, 1]):
            self._poly = geometry.polygon.Polygon(points_RZ)
            self._string = geometry.linestring.LineString(points_RZ)
            self._closed = True
        else:
            self._string = geometry.linestring.LineString(points_RZ)
            self._closed = False

    @property
    def closed(self):
        """
        True if the fluxsurface is closed.

        :return:
        """
        return self._closed

    @property
    def area(self):
        """
        Area of the closed fluxsurface.

        :return:
        """
        if self._closed:
            return self._poly.area
        else:
            raise Exception("Opened Flux Surface does not have area")

    @property
    def length(self):
        """
        Length of the fluxsurface contour

        :return:
        """
        return self._string.length

    @property
    def surface(self):
        """
        Surface of fluxsurface calculated from the contour length
        using Pappus centroid theorem : https://en.wikipedia.org/wiki/Pappus%27s_centroid_theorem

        :return: float
        """
        return self._string.length * 2 * np.pi * self.centroid.R[0]

    @property
    def centroid(self):
        return self._eq.coordinates(R=np.array(self._string.centroid.coords)[0][0],
                                    Z=np.array(self._string.centroid.coords)[0][0], coord_type=["R", "Z"])

    @property
    def volume(self):
        """
        Volume of the closed fluxsurface calculated from the area
        using Pappus centroid theorem : https://en.wikipedia.org/wiki/Pappus%27s_centroid_theorem

        :return: float
        """
        if self._closed:
            return self._poly.area * 2 * np.pi * self.centroid.R[0]
        else:
            raise Exception("Opened Flux Surface does not have area")

    @property
    def diff_volume(self):
        """
        Diferential volume :math:`V' = dV/d\psi`
        Jardin, S.: Computational Methods in Plasma Physics

        :return:
        """
        if not hasattr(self, '_diff_volume'):
            self._diff_volume = self._eval_diff_vol()

        return self._diff_volume

    def _eval_diff_vol(self):
        Rs = (self.R[1:] + self.R[:-1]) / 2
        Zs = (self.Z[1:] + self.Z[:-1]) / 2
        dpsi = self._eq.diff_psi(Rs, Zs)
        dl = self.dl

        return 2 * np.pi * np.sum(Rs * dl / dpsi)

    @property
    def dl(self):
        if not hasattr(self, '_dl'):
            self._dl = np.sqrt((self.R[1:] - self.R[:-1]) ** 2 + (self.Z[1:] - self.Z[:-1]) ** 2)
        return self._dl

class FluxSurface(Surface):
    def __init__(self, equilibrium, *coordinates, coord_type=None, grid=False, **coords):
        """
        Calculates geometrical properties of the flux surface. To make the contour closed, the first and last points in
        the passed coordinates have to be the same.
        Instance is obtained by calling method `flux_surface` in instance of `Equilibrium`.

        :param coords: Instance of coordinate class
        """

        super().__init__(equilibrium, *coordinates, coord_type=None, grid=False, **coords)

    @property
    def eval_q(self):
        if not hasattr(self, '_q'):
            self._q = self._eq.F(psi_n=np.mean(self.psi_n), grid=False) / (2 * np.pi) \
                      * self.surface_average(1 / self.R ** 2)
            # self._q = self._eq.BvacR * self.diff_volume/\
            #           (2*np.pi)**2 * self.surface_average(1/self.R**2)
        return self._q

    def get_eval_q(self, method):
        """
        :param method: str, ['sum', 'trapz', 'simps']
        :return:
        """
        return self._eq.F(psi_n=np.mean(self.psi_n), grid=False) / (2 * np.pi) \
               * self.surface_average(1 / self.R ** 2, method=method)

    @property
    def straight_fieldline_theta(self):
        r"""
        Calculate straight field line :math:`\theta^*` coordinate.

        :return:
        """
        from scipy.interpolate import CubicSpline
        from pleque.utils.tools import arglis

        if not hasattr(self, '_straight_fieldline_theta'):

            h = 1 / self.R ** 2

            izero = np.asscalar(np.argmin(np.mod(self.theta, 2 * np.pi)))

            avg = self.surface_average(h)
            cum_avg = self.cumsum_surface_average(h, roll=izero)
            # ret = np.roll(ret, amin)
            theta_star = 2 * np.pi * cum_avg / avg

            # generate splines:
            theta4spl = np.mod(self.theta, 2 * np.pi)
            th_st4spl = theta_star
            amin = np.asscalar(np.argmin(theta4spl))

            theta4spl = np.roll(theta4spl, -amin)
            th_st4spl = np.roll(th_st4spl, -amin)

            # todo: this part is for testing
            try:
                self._theta2thetastar_spl = CubicSpline(theta4spl, th_st4spl, extrapolate='periodic')
            except Exception:
                # the sequence is not always strictly increasing.
                # so get the increasing subsequence
                alis = arglis(theta4spl)

                theta4spl = theta4spl[alis]
                th_st4spl = th_st4spl[alis]
                self._theta2thetastar_spl = CubicSpline(theta4spl, th_st4spl, extrapolate='periodic')

            try:
                self._thetastar2theta_spl = CubicSpline(th_st4spl, theta4spl, extrapolate='periodic')
            except Exception:
                alis = arglis(th_st4spl)

                theta4spl = theta4spl[alis]
                th_st4spl = th_st4spl[alis]
                self._thetastar2theta_spl = CubicSpline(th_st4spl, theta4spl, extrapolate='periodic')

            self._straight_fieldline_theta = theta_star

        return self._straight_fieldline_theta

    @property
    def tor_current(self):
        """
        Return toroidal current through the closed flux surface

        :return:
        """
        if not hasattr(self, '_tor_current'):
            self._tor_current = self._eval_tor_current()
        return self._tor_current

    def _eval_tor_current(self):
        """
        to be tested (!)

        :return:
        """
        from scipy.constants import mu_0

        diff_psi = self._eq.diff_psi(self.R, self.Z)

        return 1 / mu_0 * self.surface_average(diff_psi ** 2 / self.R ** 2)

    @property
    @deprecated('Useless, will be removed. Use `abc` instead of `abc.contour`.')
    def contour(self):
        """
        Depracated. Fluxsurface contour points.
        :return: numpy ndarray
        """
        return self

    def cumsum_surface_average(self, func, roll=0):
        r"""
        Return the surface average (over single magnetic surface) value of `func`.
        Return the value of integration

        .. math::
          <func>(\psi)_i = \oint_0^{\theta_i} \frac{\mathrm{d}l R}{|\nabla \psi|}a(R, Z)

        :param func: func(X, Y), Union[ndarray, int, float]
        :return: ndarray
        """
        import inspect

        Rs = (self.R[1:] + self.R[:-1]) / 2
        Zs = (self.Z[1:] + self.Z[:-1]) / 2

        diff_psi = self._eq.diff_psi(Rs, Zs)

        if inspect.isclass(func) or inspect.isfunction(func):
            func_val = func(Rs, Zs)
        elif isinstance(func, float) or isinstance(func, int):
            func_val = func
        else:
            func_val = (func[1:] + func[:-1]) / 2

        val = self.dl * Rs / diff_psi * func_val

        val = np.roll(val, -roll)
        ret = np.cumsum(val)
        ret = np.hstack((0, ret))
        ret = np.roll(ret, +roll)

        return ret

    def surface_average(self, func, method='sum'):
        r"""
        Return the surface average (over single magnetic surface) value of `func`.
        Return the value of integration

        .. math::
          <func>(\psi) = \oint \frac{\mathrm{d}l R}{|\nabla \psi|}a(R, Z)

        :param func: func(X, Y), Union[ndarray, int, float]
        :param method: str, ['sum', 'trapz', 'simps']
        :return: 
        """
        import inspect
        from scipy.integrate import trapz, simps

        if method == 'sum':
            Rs = (self.R[1:] + self.R[:-1]) / 2
            Zs = (self.Z[1:] + self.Z[:-1]) / 2

        else:
            Rs = self.R
            Zs = self.Z

        diff_psi = self._eq.diff_psi(Rs, Zs)

        if inspect.isclass(func) or inspect.isfunction(func):
            func_val = func(Rs, Zs)
        elif isinstance(func, float) or isinstance(func, int):
            func_val = func
        else:
            if method == 'sum':
                func_val = (func[1:] + func[:-1]) / 2
            else:
                func_val = func

        l = np.hstack((0, np.cumsum(self.dl)))

        if method == 'sum':
            ret = np.sum(self.dl * Rs / diff_psi * func_val)
        elif method == 'trapz':
            ret = trapz(Rs / diff_psi * func_val, l)
        elif method == 'simps':
            ret = simps(Rs / diff_psi * func_val, l)
        else:
            ret = None

        return ret

    def contains(self, coords: Coordinates):
        points_RZ = coords.as_array(('R', 'Z'))[0, :]
        if self._closed:
            pnt = geometry.point.Point(points_RZ)
            return self._poly.contains(pnt)
        else:
            raise Exception("Opened Flux Surface does not have area")

    def distance(self, coords: Coordinates):
        point = geometry.Point(coords.as_array()[0])
        distance = self._string.distance(point)
        return distance
