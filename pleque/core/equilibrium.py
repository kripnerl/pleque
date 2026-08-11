import copy
import logging
from collections.abc import Sequence

import numpy as np
import xarray
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from shapely import Point, Polygon

import pleque.utils.equi_tools as eq_tools
import pleque.utils.flux_expansions as flux_expansion
import pleque.utils.surfaces as surf
from pleque.config.settings import get_settings
from pleque.core import Coordinates, FluxFunctions, Surface, SurfaceFunctions  # , FluxSurface
from pleque.core import cocos as cc
from pleque.core.coordinates import COORD_PARAMS_DOC, COORDINATES_DOC
from pleque.utils.decorators import (
    append_to_doc,
    deprecated,
    ordered_path_scalar_function,
    scalar_function,
    vector_function,
)
from pleque.utils.surfaces import track_plasma_boundary
from pleque.utils.tools import arglis, xp_sections

logger = logging.getLogger(__name__)


class Equilibrium:
    """
    Tokamak plasma equilibrium based on the poloidal flux function psi(R, Z).

    The equilibrium is initialised from an `xarray.Dataset` with psi on a
    rectangular (R, Z) grid and 1D profiles of the toroidal field function F
    and pressure (or their psi-derivatives) as functions of psi_n. The
    initialisation proceeds in the following steps:

    1. *Input parsing.* Spatial data, profiles and optional hints are read from
       the dataset and constructor arguments. If no first wall is given, an
       artificial rectangular wall slightly inset with respect to the psi grid
       is synthesized.
    2. *2D spline construction.* psi(R, Z) is interpolated by a
       `RectBivariateSpline` (order and smoothing from the PLEQUE settings).
    3. *Critical points.* Candidate extremes of |grad psi|^2 are located on a
       regular grid and classified by the determinant of the Hessian of psi
       into O-points (det > 0) and X-points (det < 0); the magnetic axis and
       the relevant X-points are then recognized among the candidates and the
       plasma configuration (limiter vs. X-point/diverted) is determined,
       including the limiter point and psi on the last closed flux surface.
    4. *Plasma boundary.* Strike points are found as intersections of the
       LCFS-psi contour with the first wall. The LCFS contour itself is found
       by the marching-squares algorithm on a search grid and refined by an
       iterative downhill (gradient-step) method until the relative psi error
       drops below the `lcfs.refinement_tolerance` setting; for diverted
       plasmas the X-point is inserted into the contour. (Note: poloidal
       field-line tracing of the boundary is *not* used during the
       initialisation; it is available separately via `lcfs_field_line`.)
    5. *1D profile splines.* If derivatives (pprime, FFprime) are provided,
       the p and F profiles are obtained by integration — this requires
       psi_axis and psi_lcfs, which is why this step runs *after* the
       critical-point search. Otherwise the derivatives are computed from the
       provided p and F profiles. If neither pressure nor F is available a
       vacuum equilibrium (p = F = 0) is constructed.
    6. *Midplane mapping.* A 1D spline mapping the outer-midplane radius to
       psi is built.

    Known limitations: COCOS handling is not complete (`_Bpol_sign` is not yet
    resolved from the input convention).
    """

    @staticmethod
    def _shape_spline_result(value, grid=False):
        """
        Convert SciPy spline grid output to PLEQUE's public grid layout.

        RectBivariateSpline returns true grids as (n_R, n_Z). PLEQUE exposes
        grid-shaped data as (n_Z, n_R), matching np.meshgrid(R, Z). For
        grid=False SciPy evaluates elementwise and already preserves the input
        array shape, so no transpose is applied.
        """
        return value.T if grid else value

    def __init__(self,
                 basedata: xarray.Dataset,
                 first_wall=None,
                 mg_axis=None,
                 psi_lcfs=None,
                 x_points=None,
                 strike_points=None,
                 init_method="hints",
                 spline_order=None,
                 spline_smooth=None,
                 find_extremes_order=None,
                 cocos=None,
                 verbose=False,
                 ):
        """
        Equilibrium class instance should be obtained generally by functions in pleque.io
        package.

        The hint arguments (`mg_axis`, `psi_lcfs`, `x_points`, `strike_points`) can be used
        to speed up the initialisation or to make it more robust; each of them may
        alternatively be provided as a `basedata` variable of the same name (an explicitly
        passed argument takes precedence, with a logged warning). How the hints are used is
        controlled by `init_method`.

        If the initialisation fails, the exception is logged (and re-raised); set the
        `debug_plots` PLEQUE setting (e.g. ``PLEQUE_DEBUG_PLOTS=1``) to also draw a debug
        plot of the partially initialised equilibrium.

        :param basedata: xarray.Dataset with psi(R, Z) on a rectangular R, Z grid, F(psi_n), p(psi_n)
                         (or their derivatives FFprime(psi_n), pprime(psi_n)); F = B_tor * R
        :param first_wall: array-like (Nwall, 2)  required for initialization in case of limiter configuration.
                           If not given (nor present in basedata), an artificial rectangular wall slightly
                           inset with respect to the psi grid is synthesized.
        :param mg_axis: array-like (2) suspected (R, Z) position of the magnetic axis
        :param psi_lcfs: float, suspected value of psi on the last closed flux surface
        :param x_points: array-like (n, 2) suspected (R, Z) positions of the X-points
        :param strike_points: array-like (n, 2) suspected (R, Z) positions of the strike points
        :param init_method: str One of ("full", "hints", "fast_forward"):
                            "full" — all hints are ignored and the module recognizes all critical
                            points itself;
                            "hints" (default) — the given hints are used to assist the recognition;
                            "fast_forward" — the `mg_axis` and `x_points` hints are taken as final
                            and the critical-point search is skipped (`psi_lcfs` and `strike_points`
                            hints are likewise used as final when given); if the required hints are
                            missing, the method falls back to "hints" with a logged warning.
        :param spline_order: Order of the 2D psi spline. Defaults to the value from PLEQUE settings.
        :param spline_smooth: Smoothing factor of the 2D psi spline. Defaults to the value from PLEQUE settings.
        :param find_extremes_order: Define number of points on internal grid used for identifying
                                    magnetic axis and x-points. Defaults to the value from PLEQUE settings.
        :param cocos: At the moment module assume cocos to be 3 (no other option). The implemetnation is not fully
                      working. Be aware of signs in the module!
        :param verbose: Superseded by logging: `verbose=True` calls
                        ``pleque.set_log_level(logging.DEBUG)`` (process-global). Prefer configuring
                        the ``pleque`` logger directly.
        """

        if verbose:
            # Backward-compatible alias for configuring the `pleque` logger.
            from pleque import set_log_level

            set_log_level(logging.DEBUG)

        logger.debug('Equilibrium module initialization')

        cfg = get_settings()
        if spline_order is None:
            spline_order = cfg.splines.psi_order
        if spline_smooth is None:
            spline_smooth = cfg.splines.psi_smooth
        if find_extremes_order is None:
            find_extremes_order = cfg.critical_points.find_extremes_order

        if cocos is None:
            if "cocos" in basedata.attrs:
                cocos = basedata.attrs["cocos"]
            elif "cocos" in basedata:
                cocos = basedata["cocos"]
            else:
                cocos = cfg.default_cocos

        if init_method == "fast":
            init_method = "fast_forward"
        if init_method not in ("full", "hints", "fast_forward"):
            raise ValueError(f"Unknown init_method {init_method!r}; expected one of ('full', 'hints', 'fast_forward').")

        self._basedata = basedata
        self._verbose = verbose
        self._load_hints(basedata, mg_axis, psi_lcfs, x_points, strike_points)
        self._spline_order = spline_order
        self._init_method = init_method
        self._cocos = cocos
        self._cocosdic = cc.cocos_coefs(cocos)

        # todo: resolve this from the input COCOS convention
        self._Bpol_sign = 1

        logger.debug("Equilibrium initialized with cocos=%s and init_method=%s", self._cocos, self._init_method)

        self._flux_surfaces = None

        try:
            r, z, psi = self._load_spatial_data(basedata, first_wall)
            psi_n, pressure, pprime, F, FFprime = self._load_profiles(basedata)

            logger.debug('Generating 2D spline')
            self._setup_psi_spline(r, z, psi, spline_order, spline_smooth)

            logger.debug('Looking for critical points')
            self._find_critical_points(r, z, find_extremes_order)

            logger.debug('Recognizing equilibrium type')
            self._setup_plasma_boundary()

            if 'time_unit' in basedata:
                # basedata may be an xr.Dataset, so normalise to a plain str
                # (e.g. from a 0-d DataArray) rather than storing an array.
                self.time_unit = str(np.asarray(basedata['time_unit']).item())
            else:
                self.time_unit = "ms"
            logger.debug('Generating 1D splines')
            self._setup_1d_profiles(psi_n, F, FFprime, pressure, pprime)

            logger.debug('Mapping midplane to psi_n')
            self._map_midplane2psi()

        except Exception:
            logger.exception("Equilibrium initialization failed.")

            if get_settings().debug_plots:
                try:
                    import matplotlib.pyplot as plt

                    from pleque.utils.plotting import _plot_debug

                    plt.figure()
                    _plot_debug(self)
                    plt.show()
                except Exception:
                    logger.exception("Debug plot of the failed initialization could not be drawn.")

            raise

    def _load_hints(self, basedata: xarray.Dataset, mg_axis, psi_lcfs, x_points, strike_points):
        """
        Resolve initialization hints from constructor arguments with fallback to `basedata` variables.

        An explicitly passed argument takes precedence over a `basedata` variable of the same
        name; a warning is logged when both are given.
        """

        def resolve(name, value):
            if value is not None:
                if name in basedata:
                    logger.warning("%s specified both in basedata and as an argument. Using the argument value.", name)
                return value
            if name in basedata:
                return basedata[name].values
            return None

        self._mg_axis = resolve('mg_axis', mg_axis)
        self._psi_lcfs = resolve('psi_lcfs', psi_lcfs)
        self._x_points = resolve('x_points', x_points)
        self._strike_points = resolve('strike_points', strike_points)

    def _resolve_first_wall(self, basedata: xarray.Dataset, first_wall, r, z):
        """
        Resolve the first wall from the argument, `basedata` variables, or synthesize one.

        NaN rows are dropped from the result.
        """
        if first_wall is not None:
            logger.debug("First wall taken from the constructor argument.")
        elif 'first_wall' in basedata:
            first_wall = basedata["first_wall"].values
            logger.debug("First wall taken from the 'first_wall' basedata variable.")
        elif 'R_first_wall' in basedata and 'Z_first_wall' in basedata:
            first_wall = np.array([basedata.R_first_wall.values,
                                   basedata.Z_first_wall.values]).T
            logger.debug("First wall taken from the 'R_first_wall'/'Z_first_wall' basedata variables.")
        elif 'r_lim' in basedata and 'z_lim' in basedata:
            first_wall = np.array([basedata.r_lim.values,
                                   basedata.z_lim.values]).T
            logger.debug("First wall taken from the 'r_lim'/'z_lim' basedata variables.")
        else:
            first_wall = eq_tools.synthesize_rectangular_wall(r, z)
            logger.info("No first wall given; a rectangular wall around the psi grid was synthesized.")

        first_wall = np.asarray(first_wall)
        return first_wall[~np.isnan(first_wall).any(axis=1)]

    def _load_spatial_data(self, basedata: xarray.Dataset, first_wall) -> tuple:
        """Load psi grid, first wall, and spatial metadata from dataset."""
        r = basedata.R.values
        z = basedata.Z.values
        psi = basedata.psi.transpose('R', 'Z').values

        self._first_wall = self._resolve_first_wall(basedata, first_wall, r, z)

        if 'time' in basedata:
            self.time = basedata['time'].values
        else:
            self.time = -1

        if 'time_unit' in basedata:
            self.time_unit = basedata['time_unit']
        else:
            self.time_unit = "ms"

        if 'shot' in basedata:
            self.shot = int(basedata['shot'])
        else:
            self.shot = 0

        self.R_min = np.min(r)
        self.R_max = np.max(r)
        self.Z_min = np.min(z)
        self.Z_max = np.max(z)

        return r, z, psi

    def _load_profiles(self, basedata: xarray.Dataset) -> tuple:
        """Load 1D profile arrays (F, FFprime, pressure, pprime) from dataset."""
        # TODO: allow FFprime, ffprime, pprime and other on the input
        psi_n = basedata.psi_n.values

        pressure = None
        pprime = None

        if 'pprime' in basedata:
            pprime = basedata.pprime.values
        if 'pressure' in basedata:
            pressure = basedata.pressure.values

        self.F0 = None
        if 'F0' in basedata:
            self.F0 = basedata['F0']
            if isinstance(self.F0, xarray.DataArray):
                self.F0 = np.asarray(self.F0.values).item()
        elif 'F0' in basedata.attrs:
            self.F0 = basedata.attrs['F0']

        F = None
        FFprime = None

        if 'FFprime' in basedata:
            FFprime = basedata.FFprime.values
        if 'F' in basedata:
            F = basedata.F.values

        if self.F0 is None:
            if F is not None:
                self.F0 = F[-1]
            elif 'B0' in basedata and 'R0' in basedata:
                self.F0 = basedata['B0'] * basedata['R0']
            elif 'B0' in basedata.attrs and 'R0' in basedata.attrs:
                self.F0 = basedata.attrs['B0'] * basedata.attrs['R0']

        return psi_n, pressure, pprime, F, FFprime

    def _setup_psi_spline(self, r, z, psi, spline_order: int, spline_smooth: float):
        """Build the 2D psi(R, Z) spline."""
        self._spl_psi = RectBivariateSpline(r, z, psi, kx=spline_order, ky=spline_order,
                                            s=spline_smooth)

    def _find_critical_points(self, r, z, find_extremes_order: int):
        """
        Find and classify all critical points; set plasma type and psi at LCFS.

        Hint usage depends on `init_method`: "full" ignores all hints, "hints"
        (default) uses them to assist the recognition, and "fast_forward" takes
        the magnetic-axis and X-point hints as final, skipping the critical-point
        search entirely.
        """
        use_hints = self._init_method in ("hints", "fast_forward")
        mg_axis_hint = self._mg_axis if use_hints else None
        psi_lcfs_hint = self._psi_lcfs if use_hints else None
        x_points_hint = self._x_points if use_hints else None

        trust_hints = self._init_method == "fast_forward"
        if trust_hints and (mg_axis_hint is None or x_points_hint is None):
            logger.warning("init_method='fast_forward' requires mg_axis and x_points hints; "
                           "falling back to 'hints' behaviour.")
            trust_hints = False

        if trust_hints:
            self._mg_axis = np.asarray(mg_axis_hint, dtype=float)
            self._psi_axis = np.asarray(self._spl_psi(self._mg_axis[0], self._mg_axis[1], grid=False)).item()
            self._o_points = self._mg_axis[np.newaxis, :]

            self._x_points = np.asarray(x_points_hint, dtype=float).reshape(-1, 2)
            xp1 = self._x_points[0] if len(self._x_points) > 0 else None
            self._x_point = xp1
            self._x_point2 = self._x_points[1] if len(self._x_points) > 1 else None
            self._psi_xp = self._spl_psi(*xp1, grid=False) if xp1 is not None else None
        else:
            x_points, o_points = eq_tools.find_extremes(r, z, self._spl_psi, order=find_extremes_order)

            r_lim = (self.R_min, self.R_max)
            z_lim = (self.Z_min, self.Z_max)

            self._mg_axis, sortidx = eq_tools.recognize_mg_axis(o_points, self._spl_psi, r_lim, z_lim,
                                                                first_wall=self._first_wall,
                                                                mg_axis_candidate=mg_axis_hint)
            self._psi_axis = np.asarray(self._spl_psi(self._mg_axis[0], self._mg_axis[1], grid=False)).item()
            self._o_points = o_points[sortidx]
            self._o_points[0] = self._mg_axis

            (xp1, xp2), sortidx = eq_tools.recognize_x_points(x_points, self._mg_axis, self._psi_axis,
                                                              self._spl_psi,
                                                              r_lim, z_lim, psi_lcfs_hint, x_points_hint)

            self._x_point = xp1
            self._x_point2 = xp2
            self._psi_xp = self._spl_psi(*xp1, grid=False) if xp1 is not None else None

            self._x_points = x_points[sortidx]
            if xp1 is not None:
                self._x_points[0] = xp1
            if xp2 is not None:
                self._x_points[1] = xp2

        limiter_plasma, limiter_point = eq_tools.recognize_plasma_type(self._x_point, self._first_wall,
                                                                       self._mg_axis, self._psi_axis,
                                                                       self._spl_psi)
        self._limiter_plasma = limiter_plasma
        self._limiter_point = limiter_point

        logger.info("Limiter plasma found." if limiter_plasma else "X-point plasma found.")

        if trust_hints and psi_lcfs_hint is not None:
            self._psi_lcfs = float(np.asarray(psi_lcfs_hint).item())
        else:
            self._psi_lcfs = self._spl_psi(*limiter_point, grid=False)

    def _setup_plasma_boundary(self):
        """Find LCFS contour and strike points using the recognized plasma type."""
        lcfs_cfg = get_settings().lcfs
        nr = lcfs_cfg.search_grid_nr
        nz = lcfs_cfg.search_grid_nz
        rs = np.linspace(self.R_min, self.R_max, nr)
        zs = np.linspace(self.Z_min, self.Z_max, nz)

        if self._limiter_plasma:
            self._strike_points = self._limiter_point[np.newaxis, :]
            self._contact_point = self._limiter_point
        else:
            self._contact_point = None
            if self._init_method == "fast_forward" and self._strike_points is not None:
                self._strike_points = np.asarray(self._strike_points, dtype=float).reshape(-1, 2)
            elif len(self._first_wall) < 4:
                self._strike_points = None
            else:
                self._strike_points = eq_tools.find_strike_points(self._spl_psi, rs, zs, self._psi_lcfs,
                                                                  self._first_wall)

        logger.debug("Looking for LCFS")

        # sometimes this close_lcfs is empty - investigate!
        close_lcfs = eq_tools.find_close_lcfs(self._psi_lcfs, rs, zs, self._spl_psi,
                                              self._mg_axis, self._psi_axis)

        while surf.fluxsurf_error(self._spl_psi, close_lcfs, self._psi_lcfs) > lcfs_cfg.refinement_tolerance:
            close_lcfs = eq_tools.find_surface_step(self._spl_psi, self._psi_lcfs, close_lcfs)

        if logger.isEnabledFor(logging.INFO):
            logger.info("Relative LCFS error: %s", surf.fluxsurf_error(self._spl_psi, close_lcfs, self._psi_lcfs))

        if not self._limiter_plasma:
            close_lcfs = surf.add_xpoint(self._x_point, close_lcfs, self._mg_axis)

        self._lcfs = close_lcfs

    def _setup_1d_profiles(self, psi_n, F, FFprime, pressure, pprime):
        """Build 1D profile splines and register them in fluxfuncs."""
        if self._psi_lcfs - self._psi_axis > 0:
            self._psi_sign = +1
        else:
            self._psi_sign = -1

        Fprime = None
        if FFprime is not None:
            F = eq_tools.ffprime2f(FFprime, self._psi_axis, self._psi_lcfs, self.F0)
            F_arr = np.asarray(F)
            if np.any(F_arr == 0):
                logger.warning("F profile contains zeros; F' is set to zero there.")
            Fprime = np.divide(FFprime, F_arr, out=np.zeros_like(F_arr, dtype=float), where=F_arr != 0)

        if pprime is not None:
            pressure = eq_tools.pprime2p(pprime, self._psi_axis, self._psi_lcfs)

        self.BvacR = self.F0

        self._vacuum = False
        if pressure is None or F is None:
            pressure = np.zeros_like(psi_n)
            F = np.zeros_like(psi_n)
            self._vacuum = True

        spl_cfg = get_settings().splines
        self._fpol_spl = UnivariateSpline(psi_n, F, k=spl_cfg.profile_order, s=spl_cfg.profile_smooth)

        if FFprime is None:
            self._df_dpsin_spl = self._fpol_spl.derivative()
            Fprime = self._df_dpsin_spl(psi_n) / (self._psi_lcfs - self._psi_axis)
            FFprime = F * Fprime

        self._pressure_spl = UnivariateSpline(psi_n, pressure, k=spl_cfg.profile_order, s=spl_cfg.profile_smooth)

        if pprime is None:
            self._dp_dpsin_spl = self._pressure_spl.derivative()
            pprime = self._dp_dpsin_spl(psi_n) / (self._psi_lcfs - self._psi_axis)

        self._pprime_spl = UnivariateSpline(psi_n, pprime, k=spl_cfg.profile_order, s=spl_cfg.profile_smooth)
        self._Fprime_spl = UnivariateSpline(psi_n, Fprime, k=spl_cfg.profile_order, s=spl_cfg.profile_smooth)
        self._FFprime_spl = UnivariateSpline(psi_n, FFprime, k=spl_cfg.profile_order, s=spl_cfg.profile_smooth)

        self.fluxfuncs.add_flux_func('F', F, psi_n=psi_n)
        self.fluxfuncs.add_flux_func('FFprime', FFprime, psi_n=psi_n)
        self.fluxfuncs.add_flux_func('pressure', pressure, psi_n=psi_n)
        self.fluxfuncs.add_flux_func('pprime', pprime, psi_n=psi_n)

    @scalar_function
    def psi(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        """
        Psi value

        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return coord.psi

    @vector_function(ndim=2)
    def nabla_psi(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=False, **coords) -> np.ndarray:
        r"""
        Return the value of :math:`\nabla \psi`.

        :return: Array of shape (2, ...) containing the gradient components [dψ/dR, dψ/dZ].
                If grid=True, the shape will be (2, nZ, nR).
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        dpsi_dr = self._shape_spline_result(self._spl_psi(coord.R, coord.Z, grid=coord.grid, dx=1), coord.grid)
        dpsi_dz = self._shape_spline_result(self._spl_psi(coord.R, coord.Z, grid=coord.grid, dy=1), coord.grid)

        nabla_psi = np.stack((dpsi_dr, dpsi_dz))

        return nabla_psi


    @scalar_function
    def diff_psi(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=False, **coords):
        r"""
        Return the value of :math:`\pm|\nabla \psi|`. It is positive/negative if the :math:`\psi` is increasing/decreasing.

        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        ret = np.sqrt(self._spl_psi(coord.R, coord.Z, grid=coord.grid, dx=1) ** 2 +
                      self._spl_psi(coord.R, coord.Z, grid=coord.grid, dy=1) ** 2)
        ret *= np.sign(self._psi_lcfs - self._psi_axis)
        if coord.grid:
            ret = ret.T
        return ret

    @scalar_function
    def psi_n(self, *coordinates, R=None, Z=None, psi=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi=psi, coord_type=coord_type, grid=grid, **coords)
        return coord.psi_n

    @scalar_function
    def r_mid(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        # todo: map the r_mid to psi_n
        return self._rmid_spl(coord.psi)

    @property
    def _diff_psi_n(self):
        """
        psi_2 - psi_1 = (psi_n_2 - psi_n_1)*1/_diff_psi_n
        :return: Scaling parameter between psi_n and psi
        """
        return 1 / (self._psi_lcfs - self._psi_axis)

    @scalar_function
    def rho(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return np.sqrt(coord.psi_n)

    @scalar_function
    def pressure(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._pressure_spl(coord.psi_n)

    @scalar_function
    def pprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._pprime_spl(coord.psi_n)

    @scalar_function
    def f(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)

        return self.F(coord) / mu_0

    @scalar_function
    def F(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        # todo use in_plasma
        mask_out = coord.psi_n > 1
        F = self._fpol_spl(coord.psi_n)
        F[mask_out] = self.BvacR
        return F

    @scalar_function
    def Fprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        '''

        :return:
        '''

        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        mask_out = coord.psi_n > 1
        Fprime = self._Fprime_spl(coord.psi_n)
        Fprime[mask_out] = 0
        return Fprime

    @scalar_function
    def ffprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)

        return self.FFprime(coord) / mu_0 ** 2

    @scalar_function
    def FFprime(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        mask_out = coord.psi_n > 1
        FFprime = self._fpol_spl(coord.psi_n) * self._Fprime_spl(coord.psi_n)
        FFprime[mask_out] = 0
        return FFprime

    @scalar_function
    def B_abs(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Absolute value of magnetic field in Tesla.

        :return: Absolute value of magnetic field in Tesla.
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        B_R = self.B_R(coord)
        B_Z = self.B_Z(coord)
        B_T = self.B_tor(coord)
        B_abs = np.sqrt(B_R ** 2 + B_Z ** 2 + B_T ** 2)

        return B_abs

    @vector_function(ndim=3)
    def Bvec(self, *coordinates, swap_order=False, R=None, Z=None, coord_type=None, grid=True, **coords):
        """ Magnetic field vector

        :param swap_order: If False, return component-first shape ``(3, ...)``.
                           If True, move the component axis to the end for compatibility.
        :return: Magnetic field vector array. Component-first shape is ``(3, n_elements)``
                 for paired points and ``(3, n_z, n_r)`` for grids.
        """

        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        bR = self.B_R(coord)
        bz = self.B_Z(coord)
        btor = self.B_tor(coord)

        bvec = np.stack((bR, bz, btor))

        if not swap_order:
            return bvec
        else:
            return np.moveaxis(bvec, 0, -1)

    @vector_function(ndim=3)
    def Bvec_norm(self, *coordinates, swap_order=False, R=None, Z=None, coord_type=None, grid=True, **coords):
        """ Magnetic field vector, normalised

        :param swap_order: If False, return component-first shape ``(3, ...)``.
                           If True, move the component axis to the end for compatibility.
        :return: Normalised magnetic field vector array. Component-first shape is
                 ``(3, n_elements)`` for paired points and ``(3, n_z, n_r)`` for grids.
        """

        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        bR = self.B_R(coord)
        bz = self.B_Z(coord)
        btor = self.B_tor(coord)

        bvec = np.stack((bR, bz, btor))
        bvec_n = bvec / np.linalg.norm(bvec, axis=0)

        if not swap_order:
            return bvec_n
        else:
            return np.moveaxis(bvec_n, 0, -1)

    # XXXXXX TODO TODO TODO
    @deprecated('The structure and behaviour of this function will change soon!\n'
                'to keep the same behaviour use `_flux_surface` instead.')
    def flux_surface(self, *coordinates, resolution=None, dim="step",
                     closed=True, inlcfs=True, R=None, Z=None, psi_n=None,
                     coord_type=None, **coords):
        if resolution is None:
            contour_step = get_settings().flux_surfaces.contour_step
            resolution = (contour_step, contour_step)
        return self._flux_surface(*coordinates, resolution=resolution, dim=dim,
                                  closed=closed, inlcfs=inlcfs, R=R, Z=Z, psi_n=psi_n,
                                  coord_type=coord_type, **coords)

    def _flux_surface(self, *coordinates, resolution=None, dim="step",
                      closed=None, inlcfs=True, R=None, Z=None, psi_n=None,
                      coord_type=None, **coords):
        """
        Function which finds flux surfaces with requested values of psi or psi-normalized. Specification of the
        fluxsurface properties as if it is inside last closed flux surface or if the surface is supposed to be
        closed are possible.

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
            psi_n = 1 - get_settings().lcfs.inner_psi_n_offset
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
                # generally this logic is quite odd; however, if we specify closed=None, it should add the contour
                # whatsoever
                elif not inlcfs and closed is None:
                    fluxsurface.append(contour[i])
                elif not inlcfs and (closed is True) and contour[i].closed:
                    fluxsurface.append(contour[i])
                elif not inlcfs and (closed is False) and not contour[i].closed:
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

    @scalar_function
    def poloidal_mag_flux_exp_coef(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        r"""
        **Poloidal magnetic flux expansion coefficient.**


        **Definition:**

        .. math::
           f_\mathrm{pol} = \frac{\Delta r^\mathrm{t}}{\Delta r^\mathrm{u}} =
           \frac{B_\theta^\mathrm{u} R^\mathrm{u}}{B_\theta^\mathrm{t} R^\mathrm{t}}

        **Typical usage:**

        *Poloidal magnetic flux expansion coefficient* is typically used for :math:`\lambda` scaling
        in plane perpendicular to the poloidal component of the magnetic field.

        :return:
        """

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        return flux_expansion.poloidal_mag_flux_exp_coef(self, coords)

    @ordered_path_scalar_function
    def effective_poloidal_mag_flux_exp_coef(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        r"""
        **Effective poloidal magnetic flux expansion coefficient**

        **Definition:**

        .. math::
            f_\mathrm{pol, eff} = \frac{B_\theta^\mathrm{u} R^\mathrm{u}}{B_\theta^\mathrm{t} R^\mathrm{t}}
            \frac{1}{\sin \beta} = \frac{f_\mathrm{pol}}{\sin \beta}

        Where :math:`\beta` is inclination angle of the poloidal magnetic field and the target plane.

        **Typical usage:**

        *Effective magnetic flux expansion coefficient* is typically used for :math:`\lambda` scaling
        of the target :math:`\lambda` with respect to the upstream value.

        .. math::
            \lambda^\mathrm{t} = \lambda^\mathrm{u} f_{\mathrm{pol, eff}}

        :return:
        """

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        return flux_expansion.effective_poloidal_mag_flux_exp_coef(self, coords)

    @scalar_function
    def poloidal_heat_flux_exp_coef(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        r"""
        **Poloidal heat flux expansion coefficient**

        **Definition:**

        .. math::
            f_\mathrm{pol, heat} = \frac{B_\theta^\mathrm{u}}{B_\theta^\mathrm{t}}

        **Typical usage:**
        *Poloidal heat flux expansion coefficient* is typically used to scale poloidal heat flux
        (heat flux projected along poloidal magnetic field)  along the magnetic field line.

        .. math::
            q_\theta^\mathrm{t} = \frac{q_\theta^\mathrm{u}}{f_{\mathrm{pol, heat}}}

        :return:
        """

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        return flux_expansion.poloidal_heat_flux_exp_coef(self, coords)

    @ordered_path_scalar_function
    def effective_poloidal_heat_flux_exp_coef(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        r"""
        **Effective poloidal heat flux expansion coefficient**

        **Definition:**

        .. math::
            f_\mathrm{pol, heat, eff} = \frac{B_\theta^\mathrm{u}}{B_\theta^\mathrm{t}}
            \frac{1}{\sin \beta} = \frac{f_\mathrm{pol}}{\sin \beta}

        Where :math:`\beta` is inclination angle of the poloidal magnetic field and the target plane.

        **Typical usage:**

        *Effective poloidal heat flux expansion coefficient* is typically used scale upstream poloidal
        heat flux to the target plane.

        .. math::
            q_\perp^\mathrm{t} = \frac{q_\theta^\mathrm{u}}{f_{\mathrm{pol, heat, eff}}}

        :return:
        """

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        return flux_expansion.effective_poloidal_heat_flux_exp_coef(self, coords)

    @scalar_function
    def parallel_heat_flux_exp_coef(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        r"""
        **Parallel heat flux expansion coefficient**

        **Definition:**

        .. math::
            f_\parallel= \frac{B^\mathrm{u}}{B^\mathrm{t}}

        **Typical usage:**

        *Parallel heat flux expansion coefficient* is typically used to scale total upstream heat flux
        parallel to the magnetic field along the magnetic field lines.

        .. math::
            q_\parallel^\mathrm{t} = \frac{q_\parallel^\mathrm{u}}{f_\parallel}

        :return:
        """

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        return flux_expansion.parallel_heat_flux_exp_coef(self, coords)

    @ordered_path_scalar_function
    def total_heat_flux_exp_coef(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        r"""
        **Total heat flux expansion coefficient**

        **Definition:**

        .. math::
            f_\mathrm{tot} = \frac{B^\mathrm{u}}{B^\mathrm{t}} \frac{1}{\sin \alpha} =
            \frac{f_\parallel}{\sin \alpha}

        Where :math:`\alpha` is inclination angle of the total magnetic field and the target plane.

        **Typical usage:**

        *Total heat flux expansion coefficient* is typically used to project total upstream heat flux
        parallel to the magnetic field to the target plane.

        .. math::
            q_\perp^\mathrm{t} = \frac{q_\parallel^\mathrm{u}}{f_{\mathrm{tot}}}

        :return:
        """

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        return flux_expansion.total_heat_flux_exp_coef(self, coords)

    @deprecated('This function was not written correctly and with the state of the knowlige and'
                'nobody is allowed to use it! It has been or will be replaced by the new functions.')
    def outter_parallel_fl_expansion_coef(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        WIP:Calculate parallel expansion coefitient of the given coordinates with respect to positon on the outer
        midplane.
        """
        target = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        B_midplane = self.B_abs(r=target.r_mid, theta=np.zeros_like(target.r_mid), grid=False)
        B_coord = self.B_abs(target)

        return B_coord / B_midplane

    @deprecated('This function was not written correctly and with the state of the knowlige and'
                'nobody is allowed to use it! It has been or will be replaced by the new functions.')
    def outter_poloidal_fl_expansion_coef(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        WIP:Calculate parallel expansion coefitient of the given coordinates with respect to positon on the outer
        midplane.
        """
        target = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        B_midplane = self.B_pol(r=target.r_mid, theta=np.zeros_like(target.r_mid), grid=False)
        B_coord = self.B_pol(target)

        return B_coord / B_midplane

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
            contour[i] = Coordinates.from_coords(self, contour[i])

        return contour

    def plot_geometry(self, axs=None, **kwargs):
        """
        Plots the the directions of angles, current and magnetic field.

        :param axs: None or tuple of axes. If None new figure with to axes is created.
        :param kwargs: parameters passed to the `plot` routine.
        :return: tuple of axis (ax1, ax2)
        """
        import matplotlib.pyplot as plt

        if axs is None:
            _fig, axs = plt.subplots(1, 2)

        fw = self.first_wall

        if len(fw) < 4:
            logger.warning('First wall is not sufficient. LCFS is used instead.')
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

        return axs

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

    def invert_equilibrium(self, vertically: bool = False, first_wall: bool = False, toroidal_field: bool = False,
                           current: bool = False) -> 'Equilibrium':
        """ Return a new instance of the `pleque.Equilibrium` with opposite value of selected property/quantity.

        Parameters
        ----------
        vertically : bool, optional
            If True, invert the psi map vertically (flip along Z axis). Default is False.
        first_wall : bool, optional
            If True, invert the first wall vertically. Default is False.
        toroidal_field : bool, optional
            If True, invert the toroidal field. Default is False.
        current : bool, optional
            If True, invert the current. Default is False.

        Note
        ----
        To study the effect of the inverted equilibrium to the upper divertor, to keep the (un)favourable drifts,
        one has to invert all `vertically`, `toroidal_field` and `current` quantities. `

        Returns
        -------
        pleque.Equilibrium
            A new instance of Equilibrium with inverted properties.
        """
        # Create a deep copy of the basedata
        new_basedata = copy.deepcopy(self._basedata)

        # Invert psi map vertically if requested
        if vertically:
            # Flip the psi values along the Z dimension using xarray indexing
            new_z = - new_basedata.Z.values[::-1]
            new_basedata.coords["Z"] = xarray.DataArray(new_z, dims=("Z",))
            new_psi = new_basedata.psi.transpose("R", "Z").values[:, ::-1]
            new_basedata["psi"] = xarray.DataArray(new_psi, dims=("R", "Z"))

        # Handle the first wall inversion if requested
        new_first_wall = None
        if first_wall and hasattr(self, '_first_wall') and self._first_wall is not None:
            # Create a copy of the first wall
            new_first_wall = np.copy(self._first_wall)
            # Invert the Z coordinates (second column)
            z_min = self._basedata.Z.min().item()
            z_max = self._basedata.Z.max().item()
            new_first_wall[:, 1] = z_max + z_min - new_first_wall[:, 1]

        # Invert toroidal field if requested
        if toroidal_field:
            # The toroidal field is determined by the F function and F0
            # Invert the F function in the basedata before creating the new Equilibrium
            if 'F' in new_basedata:
                new_basedata['F'] = -new_basedata['F']
            # Also invert F0 if it exists
            if 'F0' in new_basedata:
                new_basedata['F0'] = -new_basedata['F0']
            elif 'F0' in new_basedata.attrs:
                new_basedata.attrs['F0'] = -new_basedata.attrs['F0']
            # if "FFprime" in new_basedata:
            #     new_basedata["FFprime"] = -new_basedata["FFprime"]

        # Invert current if requested
        if current:
            # The current is determined by the derivatives of psi and F
            # Invert the pprime and FFprime functions if they exist
            new_basedata["psi"] = -new_basedata["psi"]
            if 'pprime' in new_basedata:
                new_basedata['pprime'] = -new_basedata['pprime']
            if 'FFprime' in new_basedata:
                new_basedata['FFprime'] = -new_basedata['FFprime']

        # Create a new Equilibrium with the modified basedata
        new_equi = Equilibrium(
            basedata=new_basedata,
            first_wall=new_first_wall if first_wall else self._first_wall,
            mg_axis=self._mg_axis,
            psi_lcfs=self._psi_lcfs,
            x_points=self._x_points,
            strike_points=self._strike_points,
            init_method=self._init_method,
            spline_order=self._spline_order,
            cocos=self._cocos,
            verbose=self._verbose
        )

        return new_equi

    def grid(self, resolution=None, dim="step"):
        """
        Function which returns 2d grid with requested step/dimensions generated over the reconstruction space.

        :param resolution: Iterable of size 2 or a number. If a number is passed,
                           R and Z dimensions will have the same size or step (depending on dim parameter). Different R and Z
                           resolutions or dimension sizes can be required by passing an iterable of size 2.
                           If None, the default grid given by the PLEQUE settings (`grid.default_nr`,
                           `grid.default_nz`) is returned.
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
                grid_cfg = get_settings().grid
                R = np.linspace(self._basedata.R.min().item(), self._basedata.R.max().item(), grid_cfg.default_nr)
                Z = np.linspace(self._basedata.Z.min().item(), self._basedata.Z.max().item(), grid_cfg.default_nz)
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
                    R = np.arange(self._basedata.R.min().item(), self._basedata.R.max().item(), res_R)
                elif dim[0] == "size":
                    R = np.linspace(self._basedata.R.min().item(), self._basedata.R.max().item(), res_Z)
                else:
                    raise ValueError("Wrong dim[0] value passed")

                if res_Z is None:
                    Z = self._basedata.Z.values
                elif dim[1] == "step":
                    Z = np.arange(self._basedata.Z.min().item(), self._basedata.R.max().item(), res_R)
                elif dim[1] == "size":
                    Z = np.linspace(self._basedata.Z.min().item(), self._basedata.Z.max().item(), res_Z)
                else:
                    raise ValueError("Wrong dim[1] value passed")
            elif isinstance(dim, str):
                if dim == "step":
                    if res_R is None:
                        R = self._basedata.R.values
                    else:
                        R = np.arange(self._basedata.R.min().item(), self._basedata.R.max().item(), res_R)
                    if res_Z is None:
                        Z = self._basedata.Z.values
                    else:
                        Z = np.arange(self._basedata.Z.min().item(), self._basedata.Z.max().item(), res_Z)
                elif dim == "size":
                    if res_R is None:
                        R = self._basedata.R.values
                    else:
                        R = np.linspace(self._basedata.R.min().item(), self._basedata.R.max().item(), res_R)
                    if res_Z is None:
                        Z = self._basedata.Z.values
                    else:
                        Z = np.linspace(self._basedata.Z.min().item(), self._basedata.Z.max().item(), res_Z)
                else:
                    raise ValueError("Wrong dim value passed")
            else:
                raise ValueError("Wrong dim value passed")

        coords = self.coordinates(R=R, Z=Z, grid=True)

        return coords

    @scalar_function
    def B_R(self, *coordinates, R=None, Z=None, coord_type=('R', 'Z'), grid=True, **coords):
        """
        Poloidal value of magnetic field in Tesla.

        :return: Scalar array with shape ``(n_elements,)`` for paired points and
                 ``(n_z, n_r)`` for grids.
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        cc_norm = self._cocosdic["sigma_cyl"] * self._cocosdic["sigma_Bp"] * 1 / (2 * np.pi) ** self._cocosdic["exp_Bp"]
        dpsi_dz = self._shape_spline_result(self._spl_psi(coord.R, coord.Z, dy=1, grid=coord.grid), coord.grid)
        R = self._shape_spline_result(coord.R, coord.grid)
        return cc_norm * dpsi_dz / R * self._Bpol_sign

    def B_R_rz(self, R, Z):
        cc_norm = self._cocosdic["sigma_cyl"] * self._cocosdic["sigma_Bp"] * 1 / (2 * np.pi) ** self._cocosdic["exp_Bp"]
        return cc_norm * self._spl_psi(R, Z, dy=1, grid=False) / R * self._Bpol_sign

    @scalar_function
    def B_Z(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Poloidal value of magnetic field in Tesla.

        :return: Scalar array with shape ``(n_elements,)`` for paired points and
                 ``(n_z, n_r)`` for grids.
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        cc_norm = self._cocosdic["sigma_cyl"] * self._cocosdic["sigma_Bp"] * 1 / (2 * np.pi) ** self._cocosdic["exp_Bp"]
        dpsi_dr = self._shape_spline_result(self._spl_psi(coord.R, coord.Z, dx=1, grid=coord.grid), coord.grid)
        R = self._shape_spline_result(coord.R, coord.grid)
        return - cc_norm * dpsi_dr / R * self._Bpol_sign

    def B_Z_rz(self, R, Z):
        cc_norm = self._cocosdic["sigma_cyl"] * self._cocosdic["sigma_Bp"] * 1 / (2 * np.pi) ** self._cocosdic["exp_Bp"]
        return - cc_norm * self._spl_psi(R, Z, dx=1, grid=False) / R * self._Bpol_sign


    @scalar_function
    def B_pol(self, *coordinates, R=None, Z=None, coord_type=None, grid=True, **coords):
        """
        Absolute value of magnetic field in Tesla.

        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        B_R = self.B_R(coord)
        B_Z = self.B_Z(coord)
        B_pol = np.sqrt(B_R ** 2 + B_Z ** 2)
        return B_pol

    @scalar_function
    def B_tor(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        """
        Toroidal value of magnetic field in Tesla.

        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return self.F(coord) / coord.R

    def B_tor_rz(self, R, Z):

        psi = self._spl_psi(R, Z, grid=False)
        psi_n = (psi - self._psi_axis) / (self._psi_lcfs - self._psi_axis)

        mask_out = psi_n > 1
        F = self._fpol_spl(psi_n)
        F[mask_out] = self.BvacR
        return F / R

    @scalar_function
    def abs_q(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        """
        Absolute value of q.

        :return:
        """
        return np.abs(self.q(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords))

    @scalar_function
    def q(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        if not hasattr(self, '_q_spl'):
            self._init_q()
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return self._q_spl(coord.psi_n)

    @scalar_function
    def diff_q(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=False, **coords):
        """

        :param self:
        :return: Derivative of q with respect to psi.
        """
        if not hasattr(self, '_dq_dpsin_spl'):
            self._init_q()
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        return self._dq_dpsin_spl(coord.psi_n) * self._diff_psi_n

    @scalar_function
    def shear(self, *coordinates, R=None, Z=None, psi_n=None, coord_type=None, grid=False, **coords):
        r"""Normalized magnetic shear parameter

        .. math::
          \hat s = \frac{r_\mathrm{mid}}{q}\frac{\mathrm{d}q}{\mathrm{d}r}

        where r_\mathrm{mid} is a plasma radius on the midplane.
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, psi_n=psi_n, coord_type=coord_type, grid=grid, **coords)
        q = self.q(coord)
        dq_dpsi = self.diff_q(coord)
        dpsi_dr = self.diff_psi(coord)
        s = coord.r_mid / q * dq_dpsi * dpsi_dr
        return s

    @scalar_function
    def pol_flux(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        """
        Return poloidal flux in Wb which is not normalized by 2pi defiend as:

        .. math::
            \\psi_\\mathrm{pol} = - \rho_{Bp} \\int B \\mathrm{d}S

        the result is obtained by normalization of the reference poloidal flux `psi`
        with respect to the current value of the COCOS:

        .. math::
            \\psi_\\mathrm{pol} = (2 \\pi)^{1 - e_{Bp}} \\psi_\\mathrm{ref}

        :return:
        """

        cc = 2 * np.pi if self._cocosdic['exp_Bp'] == 0 else 1
        return cc * self.psi(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

    @scalar_function
    def tor_flux(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        """
        Calculate toroidal magnetic flux :math:`\\Phi` from:

        .. math::
            q = \frac{\\mathrm{d \\Phi} }{\\mathrm{d \\psi}}

        :return:
        """

        if not hasattr(self, '_q_anideriv_spl'):
            self._init_q()
        cc = self._cocosdic['sigma_Bp'] * self._cocosdic['sigma_pol'] * ((2 * np.pi) ** (1 - self._cocosdic['exp_Bp']))
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return cc * self._q_anideriv_spl(coord.psi_n) * (1 / self._diff_psi_n)

    @scalar_function
    def j_R(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        # todo test cocos here!
        # todo: test grid
        # todo: test test test
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        cc = self._cocosdic['sigma_cyl']

        dpsi_dZ = self._spl_psi(coord.R, coord.Z, grid=coord.grid, dy=1)
        if coord.grid:
            dpsi_dZ = dpsi_dZ.T

        return - cc * self.Fprime(coord) / (coord.R * mu_0) * dpsi_dZ

    @scalar_function
    def j_Z(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        cc = self._cocosdic['sigma_cyl']

        dpsi_dR = self._spl_psi(coord.R, coord.Z, grid=coord.grid, dx=1)
        if coord.grid:
            dpsi_dR = dpsi_dR.T

        return cc * self.Fprime(coord) / (coord.R * mu_0) * dpsi_dR

    @scalar_function
    def j_pol(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=False, **coords):
        r"""
        Poloidal component of the current density.
        Calculated as

        .. math::
          \frac{f' \nabla \psi }{R \mu_0}

        [Wesson: Tokamaks, p. 105]

        :return:
        """
        coord = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)
        return self.Fprime(coord) / (coord.R * mu_0) * self.diff_psi(coord)

    @scalar_function
    def j_tor(self, *coordinates, R: np.array = None, Z: np.array = None, coord_type=None, grid=True, **coords):
        r"""
        todo: to be tested

        Toroidal component of the current denisity.
        Calculated as

        .. math::
          R p' + \frac{1}{\mu_0 R} ff'

        :return:
        """
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
        cnt = 0
        # todo: This should be rewritten
        while not found and cnt < 101:
            psi_n = 1 + 1e-6 * cnt
            cnt += 1
            separatrix = self._flux_surface(inlcfs=False, closed=None, psi_n=psi_n)

            for flux_surf in separatrix:
                # todo: this is not separatrix... for example in limiter plasma and without first wall
                intersection = self.first_wall.intersection(flux_surf)
                # The behaviour of intersetion function changed. This condition should catch both empty list and None:
                if intersection and len(intersection) > 0:
                    poly = Polygon(flux_surf._string)
                    o_point = Point(self.magnetic_axis.R[0], self.magnetic_axis.Z[0])
                    if poly.contains(o_point):
                        self._separatrix = flux_surf.as_array(("R", "Z"))
                        found = True
                        break

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
    def x_point(self) -> Coordinates | None:
        """
        Return x-point closest in psi to mg-axis if presented on grid. None otherwise.

        :return Coordinates
        """
        if self._x_point is None:
            return None
        else:
            return self.coordinates(*self._x_point)

    @property
    def secondary_x_point(self) -> Coordinates | None:
        """
        Returns the second closest (in units of psi) x-point to magnetic axis. None otherwise.
        """
        if self._x_point2 is None:
            return None
        else:
            return self.coordinates(*self._x_point2)

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
                first_wall = np.concatenate((first_wall, first_wall[0, :][None, :]), axis=0)
            return Surface(self, first_wall)

    @property
    def magnetic_axis(self):
        return self.coordinates(self._mg_axis[0], self._mg_axis[1])

    @property
    def geometrical_axis(self):
        """
        Geometrical axis of the plasma defined as

        :math
            R_{ax} = (max(R_{lcfs}) + min(R_{lcfs})) / 2
            Z_{ax} = (max(Z_{lcfs}) + min(Z_{lcfs})) / 2

        """

        r_ax = (self.lcfs.R.max() + self.lcfs.R.min()) / 2
        z_ax = (self.lcfs.Z.max() + self.lcfs.Z.min()) / 2

        return self.coordinates(r_ax, z_ax)

    @property
    def I_plasma(self):
        """
        Toroidal plasma current. Calculated as toroidal current through the LCFS.

        :return: (float) Value of toroidal plasma current.
        """
        if not hasattr(self, "_Ip"):
            self._Ip = self.lcfs.tor_current
        return self._Ip


    def xp_section(self, length: float = 0.15) -> tuple["Coordinates"]:
        """
        Return poloidal cross-sections of the planes of the x-point section (Σ_s).

        Args:
            length (float): Length around the X-point for computing sections. Defaults to 0.15.

        Returns:
            tuple[Coordinates]: Tuple of `Coordinates` objects representing each plane.
            The order of x-point planes directions (with respect to x-point) (lfs, in-plasma, hfs, out) is preserved.
        """
        secs = xp_sections(self._spl_psi, self._x_point[0], self._x_point[1], length=length)

        secs_coords = tuple(self.coordinates(sec) for sec in secs)
        return secs_coords

    @append_to_doc(COORDINATES_DOC)
    def coordinates(self, *coordinates, coord_type=None, grid=False, **coords):
        """
        Return instance of Coordinates. If instances of coordinates is already on the input, just pass it through.

        :param coordinates: Positional coordinates; see below.
        :param coord_type: Tuple naming the input coordinates, e.g. ``('rho',)`` or ``('Z', 'R')``.
        :param grid: If ``True``, the coordinates span a rectangular grid (2D only).
        :param coords: Coordinates passed by name; see below.
        :return: Instance of :class:`pleque.core.Coordinates`.
        """
        if len(coordinates) >= 1 and isinstance(coordinates[0], Coordinates):
            return coordinates[0]
        else:
            return Coordinates.from_coords(self, *coordinates, coord_type=coord_type, grid=grid, **coords)

    def _as_fluxsurface(self, *coordinates, coord_type=None, grid=False, **coords):
        """

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
        points = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

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
        points = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, grid=grid, **coords)

        mask_in = points_inside_curve(points.as_array(), self._lcfs)
        if points.grid:
            mask_in = mask_in.reshape(len(points.x2), len(points.x1))
        return mask_in

    def connection_length(self, *coordinates, R: np.array = None, Z: np.array = None,
                          coord_type=None, direction=1, **coords):
        """
        Calculate connection length from given coordinates to first wall

        Todo: The field line is traced to min/max value of z of first wall, distance is calculated to the last
            point before first wall.
        :param direction: if positive trace field line in/cons the direction of magnetic field.
        :param stopper: (None, 'poloidal', 'z-stopper) force to use stopper. If None stopper is
                       automatically chosen based on psi_n coordinate.
        :return:
        """
        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)
        traces = self.trace_field_line(coords, direction=direction)
        dists = []
        lines = []

        for t in traces:
            if t.psi_n[0] > 1:
                # todo: find first False!
                mask_in = self.in_first_wall(t)
                rzp = t.as_array()[mask_in, :]

                # todo: add intersection point!

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

        :param direction: if positive trace field line in/cons the direction of magnetic field.
        :param stopper_method: (None, 'poloidal', 'z-stopper) force to use stopper. If None stopper is
                       automatically chosen based on psi_n coordinate.
        :param in_first_wall: if True the only inner part of field line is returned.
        :return:

        """
        from scipy.integrate import solve_ivp

        import pleque.utils.field_line_tracers as flt

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

            flt_cfg = get_settings().field_line_tracing
            atol = flt_cfg.atol
            if self.is_xpoint_plasma:
                xp = self._x_point
                xp_dist = np.sqrt(np.sum((xp - y0) ** 2))
                atol = np.minimum(xp_dist * flt_cfg.x_point_atol_scale, atol)

            logger.debug('tracing from: %3f,%3f,%3f', y0[0], y0[1], phi0)
            logger.debug('atol = %s', atol)

            stopper = None

            if stopper_method is None:
                if coords.psi_n[i] <= 1:
                    # todo: determine the direction (now -1) !!
                    logger.debug('poloidal stopper is used')

                    # XXX Direction (TODO)
                    # XXX add these values to cocos dict!
                    # sign(dtheta/dphi) = sigma_pol * sign(I * B)
                    # dphidtheta = self._cocosdic['sigma_pol'] * np.sign(self.I_plasma) * np.sign(self.F0)

                    dphidtheta = np.sign(self.F0) * self._cocosdic['sigma_pol'] * self._cocosdic['sigma_cyl']
                    logger.debug('direction: %s', direction)
                    logger.debug('dphidtheta: %s', dphidtheta)

                    stopper = flt.poloidal_angle_stopper_factory(y0, self.magnetic_axis.as_array()[0],
                                                                 dphidtheta * direction,
                                                                 stop_res=flt_cfg.poloidal_stop_resolution)
                else:
                    logger.debug('z-lim stopper is used')
                    stopper = flt.rz_coordinate_stopper_factory(r_lims, z_lims)
            elif stopper_method == 'z-stopper':
                logger.debug('z-lim stopper is used')
                stopper = flt.rz_coordinate_stopper_factory(r_lims, z_lims)
            elif stopper_method == 'poloidal':
                logger.debug('poloidal stopper is used')

                dphidtheta = np.sign(self.F0) * self._cocosdic['sigma_pol'] * self._cocosdic['sigma_cyl']
                stopper = flt.poloidal_angle_stopper_factory(y0, self.magnetic_axis.as_array()[0],
                                                             dphidtheta * direction,
                                                             stop_res=flt_cfg.poloidal_stop_resolution)

            # todo: define somehow sufficient tolerances
            sol = solve_ivp(dphifunc,
                            (phi0, direction * sigma_B0 * (2 * np.pi * flt_cfg.max_toroidal_turns + phi0)),
                            y0,
                            #                            method='RK45',
                            method='LSODA',
                            events=stopper,
                            max_step=flt_cfg.max_step,  # we want high phi resolution
                            atol=atol,
                            rtol=flt_cfg.rtol,
                            )

            logger.debug("%s, %s", sol.message, sol.nfev)

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

    def lcfs_field_line(self, vect_no=0, xp_shift=None, phi0: float = 0.0):
        """
        Computes (some) field line laying on last closed flux surface.

        Parameters:
        vect_no: int
            Index of eigenvector determing the direction of integration. Default is 0.
        xp_shift: float
            A small positional adjustment for the x-point in the plasma boundary tracking.
            Defaults to the value from PLEQUE settings (`lcfs.x_point_shift`).
        phi0: float
            Toroidal angle on which is field line initiated.

        Returns:
        list
            A representation of the LCFS field line as a result of the plasma boundary
            tracking method.
        """
        if xp_shift is None:
            xp_shift = get_settings().lcfs.x_point_shift
        lcfs = track_plasma_boundary(self, self._x_point, vect_no=vect_no, xp_shift=xp_shift, phi_0=phi0)
        return lcfs

    def trace_flux_surface(self, *coordinates, s_resolution=None, R=None,
                           Z=None, psi_n=None, coord_type=None, **coords):
        """
        Find a closed flux surface inside LCFS with requested values of psi or psi-normalized.


        TODO support open and/or flux surfaces outise LCFS, needs different stopper

        :param coordinates: specifies flux surface to search for (by spatial point or values of psi or psi normalised).
                            If coordinates is spatial point (dim=2) then the trace starts at the midplane.
                            Coordinates.grid must be False.
        :param s_resolution: max_step in the distance along the flux surface contour.
                             Defaults to the value from PLEQUE settings (`flux_surfaces.trace_step`).
        :return: FluxSurface
        """
        from scipy.integrate import solve_ivp

        import pleque.utils.field_line_tracers as flt

        fs_cfg = get_settings().flux_surfaces
        if s_resolution is None:
            s_resolution = fs_cfg.trace_step

        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)
        if coords.dim == 1:
            coords = self.coordinates(R=self.magnetic_axis.R + coords.r_mid, Z=0)
        y0 = np.reshape([coords.R, coords.Z], (2,))

        ds_func = flt.ds_grad_psi_tracer_factory(self._spl_psi)
        # atol is square of s_resolution to offset the square distance
        stopper = flt.rz_target_s_min_stopper_factory(y0, coords.r_mid,
                                                      atol=s_resolution ** 2)

        sol = solve_ivp(ds_func,
                        (0, coords.r_mid * 2 * np.pi * fs_cfg.trace_max_turns),
                        y0,
                        method='LSODA',
                        events=stopper,
                        max_step=s_resolution,
                        vectorized=True,
                        # rtol=1e-8,
                        )
        fs = self._as_fluxsurface(R=np.hstack([sol.y[0], sol.y[0, 0]]),
                                  Z=np.hstack([sol.y[1], sol.y[1, 0]]))
        fs._cum_length = np.hstack([sol.t, sol.t[-1] + s_resolution])  # TODO use some setter
        return fs

    @property
    def fluxfuncs(self):
        if not hasattr(self, '_fluxfunc'):
            self._fluxfunc = FluxFunctions(self)  # filters out methods from self
        return self._fluxfunc

    @property
    def surfacefuncs(self):
        if not hasattr(self, '_surfacefunc'):
            self._surfacefunc = SurfaceFunctions(self)  # filters out methods from self
        return self._surfacefunc

    def to_geqdsk(self, file, nx=None, ny=None, q_positive=True, use_basedata=False, cocos_out=3):
        """
        Write a GEQDSK/g-file equilibrium file.

        :param file: str, file name
        :param nx: int, number radial points and profiles points.
                   Defaults to the value from PLEQUE settings (`io.geqdsk_nx`).
        :param ny: int, number of vertical points.
                   Defaults to the value from PLEQUE settings (`io.geqdsk_ny`).
        :param use_basedata: The original basedata of equilibrium are used instead of interpolation splines.
                             If this option is chosen, the nx and ny parameters are ignored.
        :param q_positive: Save q value always positive.
        :param cocos_out: Number of output cocos. `None` if not changed.

        Warning: `cocos_out` parameter only perform 2 pi normalization at the moment.
        Default value is 3, which corresponds to standard eqdsk EFIT output.
        """
        import pleque.io.geqdsk as geqdsk

        geqdsk.write(self, file, nx=nx, ny=ny, q_positive=q_positive, use_basedata=use_basedata,
                     cocos_out=cocos_out)

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

    def _map_midplane2psi(self):
        from scipy.interpolate import UnivariateSpline

        r_mid = np.linspace(0, self.R_max - self._mg_axis[0], get_settings().flux_surfaces.midplane_map_points)
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
        spl_cfg = get_settings().splines
        self._rmid_spl = UnivariateSpline(psi_mid, r_mid, k=spl_cfg.profile_order, s=spl_cfg.profile_smooth)

    def _init_fluxsurfaces(self, npsi: int | None = None, psi_n_levels: Sequence[float] | None = None):

        if psi_n_levels and npsi:
            raise ValueError("npsi and psi_n_levels cannot be used simultaneously.")
        if psi_n_levels is None:
            fs_cfg = get_settings().flux_surfaces
            psi0 = fs_cfg.psi_n_min
            if npsi is None:
                npsi = fs_cfg.n_psi
            psi_n_levels = np.linspace(psi0, 1, npsi)
        else:
            npsi = len(psi_n_levels)

        # todo: Now I need to rewrite find_flux_surface function first to used multiple psi levels.
        surfs = []
        for psi_n in psi_n_levels:
            surf = self.find_flux_surface(psi_n=psi_n)[0]
            surfs.append(surf)

        self._flux_surfaces = surfs

    def _init_q(self):
        fs_cfg = get_settings().flux_surfaces
        psi_n = np.arange(fs_cfg.q_psi_n_min, 1, fs_cfg.q_psi_n_step)
        qs = []

        logger.debug("Generating q-splines")
        for i, pn in enumerate(psi_n):
            if np.mod(i, 20) == 0:
                logger.debug("q-spline progress: %.0f%%", pn / np.max(psi_n) * 100)
            surface = self._flux_surface(psi_n=pn)
            c = surface[0]
            qs.append(c.eval_q)
        qs = np.array(qs)

        spl_cfg = get_settings().splines
        self._q_spl = UnivariateSpline(psi_n, qs, s=spl_cfg.profile_smooth, k=spl_cfg.profile_order)
        self._dq_dpsin_spl = self._q_spl.derivative()
        self._q_anideriv_spl = self._q_spl.antiderivative()


# All public Equilibrium methods accepting the rich coordinate input
# (see ``Coordinates.from_coords``). A note cross-referencing the canonical
# description of the input syntax is appended to their docstrings, so the
# syntax itself is documented in a single place only.
_COORDINATE_INPUT_METHODS = [
    'psi', 'nabla_psi', 'diff_psi', 'psi_n', 'r_mid', 'rho',
    'pressure', 'pprime', 'f', 'F', 'Fprime', 'ffprime', 'FFprime',
    'B_abs', 'Bvec', 'Bvec_norm', 'B_R', 'B_Z', 'B_pol', 'B_tor',
    'abs_q', 'q', 'diff_q', 'shear', 'pol_flux', 'tor_flux',
    'j_R', 'j_Z', 'j_pol', 'j_tor',
    'flux_surface',
    'poloidal_mag_flux_exp_coef', 'effective_poloidal_mag_flux_exp_coef',
    'poloidal_heat_flux_exp_coef', 'effective_poloidal_heat_flux_exp_coef',
    'parallel_heat_flux_exp_coef', 'total_heat_flux_exp_coef',
    'outter_parallel_fl_expansion_coef', 'outter_poloidal_fl_expansion_coef',
    'in_first_wall', 'in_lcfs',
    'connection_length', 'trace_field_line', 'trace_flux_surface',
]

for _name in _COORDINATE_INPUT_METHODS:
    append_to_doc(COORD_PARAMS_DOC)(getattr(Equilibrium, _name))
del _name
