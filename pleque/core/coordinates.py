import logging
import warnings
from collections.abc import Iterable
from typing import ClassVar, Union

import numpy as np
from scipy.interpolate import splev, splprep

import pleque.utils.flux_expansions as flux_expansion
from pleque.utils.decorators import append_to_doc, deprecated

from .cocos import cocos_coefs

logger = logging.getLogger(__name__)

#: Canonical description of the coordinate input syntax accepted across PLEQUE.
#: It is appended (via :func:`pleque.utils.decorators.append_to_doc`) to the
#: docstrings of :meth:`Coordinates.from_coords` and
#: :meth:`pleque.core.Equilibrium.coordinates`, so it is written only once.
COORDINATES_DOC = r"""
**Coordinate input syntax**

Coordinates may be specified positionally or by name:

* nothing -- an empty (``dim = 0``) object is created,
* a single ``array (N, dim)`` -- ``N`` points will be generated,
* one, two or three comma separated one dimensional arrays or scalars,
* named coordinates: ``R=, Z=, psi_n=, psi=, rho=, r=, theta=, phi=, X=, Y=``.

**Default coordinate systems**

- **1D**: :math:`\psi_\mathrm{N}`,
- **2D**: :math:`(R, Z)`,
- **3D**: :math:`(R, Z, \phi)`.

**Accepted coordinates types**

*1D - coordinates*

+------------------------+-----------+------------------------------+
| Coordinate             | Code      | Note                         |
+========================+===========+==============================+
|:math:`\psi_\mathrm{N}` | ``psi_n`` | Default 1D coordinate        |
+------------------------+-----------+------------------------------+
|:math:`\psi`            | ``psi``   |                              |
+------------------------+-----------+------------------------------+
|:math:`\rho`            | ``rho``   | :math:`\rho = \sqrt{\psi_n}` |
+------------------------+-----------+------------------------------+

*2D - coordinates*

+------------------------+--------------+-------------------------------------------------+
| Coordinate             | Code         | Note                                            |
+========================+==============+=================================================+
|:math:`(R, Z)`          | ``R, Z``     | Default 2D coordinate                           |
+------------------------+--------------+-------------------------------------------------+
|:math:`(r, \theta)`     | ``r, theta`` | Polar coordinates with respect to magnetic axis |
+------------------------+--------------+-------------------------------------------------+

*3D - coordinates*

+------------------------+---------------+-------------------------------------------------+
| Coordinate             | Code          | Note                                            |
+========================+===============+=================================================+
|:math:`(R, Z, \phi)`    | ``R, Z, phi`` | Default 3D coordinate                           |
+------------------------+---------------+-------------------------------------------------+
|:math:`(X, Y, Z)`       | ``(X, Y, Z)`` | Cartesian coordinates                           |
+------------------------+---------------+-------------------------------------------------+
"""

#: Short note cross-referencing the canonical coordinate syntax description.
#: Appended to the docstrings of all coordinate-accepting ``Equilibrium``
#: methods.
COORD_PARAMS_DOC = """
.. note:: This method accepts PLEQUE's rich coordinate input: an existing
   :class:`~pleque.core.coordinates.Coordinates` instance, positional
   values (``R, Z`` by default for 2D input), or coordinates passed by name
   (``R=``, ``Z=``, ``psi_n=``, ``psi=``, ``rho=``, ...), together with the
   ``coord_type`` and ``grid`` options. See
   :meth:`pleque.core.coordinates.Coordinates.from_coords` for the full
   description of the syntax.
"""


class Coordinates:
    r"""
    Basic PLEQUE class to handle various coordinate systems in tokamak
    equilibrium.

    Preferably created with :meth:`Coordinates.from_coords` or
    :meth:`pleque.core.Equilibrium.coordinates`, which accept the full
    coordinate input syntax described in :meth:`Coordinates.from_coords`.
    """

    # Sets of recognised coordinate names, shared by all instances:
    _valid_coordinates: ClassVar[set] = {'R', 'Z', 'psi_n', 'psi', 'rho', 'r', 'theta', 'phi', 'X', 'Y'}
    _valid_coordinates_1d: ClassVar[set] = {('psi_n',), ('psi',), ('rho',)}
    _valid_coordinates_2d: ClassVar[set] = {('R', 'Z'), ('r', 'theta')}
    _valid_coordinates_3d: ClassVar[set] = {('R', 'Z', 'phi'), ('X', 'Y', 'Z')}

    # Default coordinate type for a given dimension:
    _default_coord_types: ClassVar[dict] = {1: ('psi_n',), 2: ('R', 'Z'), 3: ('R', 'Z', 'phi')}

    def __init__(self, equilibrium, *coordinates, coord_type=None, grid=False, cocos=None, **coords):
        r"""
        Create Coordinates from positional scalars or 1-D arrays in the
        default coordinate type of the given dimension (``psi_n``; ``R, Z``;
        ``R, Z, phi``).

        .. deprecated:: 0.0.11
            Constructing ``Coordinates`` directly from the rich coordinate
            input (named coordinates such as ``R=``/``psi_n=``,
            ``array (N, dim)`` input or a non-default ``coord_type``) is
            deprecated; use :meth:`Coordinates.from_coords` or
            :meth:`pleque.core.Equilibrium.coordinates` instead.

        :param equilibrium: Instance of ``Equilibrium`` or ``None``.
        :param coordinates: One, two or three comma separated scalars or
                            one dimensional arrays.
        :param coord_type: Tuple naming the input coordinates. Only the
                           default coordinate type of the corresponding
                           dimension is accepted without deprecation warning.
        :param grid: If ``True``, the coordinates span a rectangular grid
                     (2D only).
        :param cocos: Define coordinate system cocos. If `None` equilibrium default cocos is used.
                        If `equilibrium is None` cocos = 3  (both systems cnt-clockwise) is used.
        :param coords: Deprecated here; coordinates passed by name. Use
                       :meth:`Coordinates.from_coords` instead.
        """
        if self._is_rich_input(coordinates, coord_type, coords):
            warnings.warn(
                'Constructing Coordinates directly from the rich coordinate input '
                '(named coordinates, array (N, dim) input or a non-default coord_type) '
                'is deprecated; use Coordinates.from_coords() or '
                'Equilibrium.coordinates() instead.',
                DeprecationWarning,
                stacklevel=2,
            )

        self._init_from_input(equilibrium, coordinates, coord_type, grid, cocos, coords)

    @classmethod
    @append_to_doc(COORDINATES_DOC)
    def from_coords(cls, equilibrium, *coordinates, coord_type=None, grid=False, cocos=None, **coords):
        r"""
        Create :class:`Coordinates` from any supported coordinate
        specification.

        This is the preferred way of constructing :class:`Coordinates` and
        the canonical description of the coordinate input syntax accepted
        across PLEQUE (for instance by most
        :class:`pleque.core.Equilibrium` methods).

        :param equilibrium: Instance of ``Equilibrium`` or ``None``.
        :param coordinates: Positional coordinates; see below.
        :param coord_type: Tuple naming the input coordinates,
                           e.g. ``('rho',)`` or ``('Z', 'R')``.
        :param grid: If ``True``, the coordinates span a rectangular grid
                     (2D only).
        :param cocos: Define coordinate system cocos. If `None` equilibrium default cocos is used.
                        If `equilibrium is None` cocos = 3  (both systems cnt-clockwise) is used.
        :param coords: Coordinates passed by name; see below.
        :return: Instance of :class:`Coordinates`.
        """
        if cls is Coordinates:
            self = object.__new__(cls)
            self._init_from_input(equilibrium, coordinates, coord_type, grid, cocos, coords)
            return self
        # Subclasses (e.g. Surface) run additional logic in __init__:
        return cls(equilibrium, *coordinates, coord_type=coord_type, grid=grid, cocos=cocos, **coords)

    @classmethod
    def _is_rich_input(cls, coordinates, coord_type, coords):
        """Return True when the input requires the full (deprecated in
        ``__init__``) rich-syntax parser instead of positional values in the
        default coordinate type."""
        if any(v is not None for v in coords.values()):
            return True
        n = len(coordinates)
        if n == 0:
            return False
        if coord_type is not None:
            if isinstance(coord_type, str):
                coord_type = (coord_type,)
            if tuple(coord_type) != cls._default_coord_types.get(n):
                return True
        return n == 1 and np.ndim(coordinates[0]) > 1

    def _init_from_input(self, equilibrium, coordinates, coord_type, grid, cocos, coords):
        """Initialise the instance from any supported coordinate
        specification (without any deprecation warning)."""
        self._eq = equilibrium
        self.dim = -1  # init only
        self.grid = grid

        if cocos is None:
            if equilibrium is None:
                self.cocos = 3
            else:
                self.cocos = equilibrium.cocos
        else:
            self.cocos = cocos

        self.cocos_dict = cocos_coefs(self.cocos)

        if self._init_fast_path(coordinates, coord_type, coords):
            return

        self._evaluate_input(*coordinates, coord_type=coord_type, **coords)

    def _init_fast_path(self, coordinates, coord_type, coords):
        """
        Set the coordinate attributes directly for the most common input:
        one to three positional scalars or 1-D arrays already in the default
        coordinate type. Return True on success; on False the caller falls
        back to the full input parser.
        """
        n = len(coordinates)
        if not 1 <= n <= 3:
            return False
        if coords and not all(v is None for v in coords.values()):
            return False
        default_type = self._default_coord_types[n]
        if coord_type is not None:
            if isinstance(coord_type, str):
                coord_type = (coord_type,)
            if tuple(coord_type) != default_type:
                return False

        xs = []
        for c in coordinates:
            if isinstance(c, np.ndarray):
                if c.ndim != 1 or not np.issubdtype(c.dtype, np.number):
                    return False
                xs.append(c)
            elif isinstance(c, (float, int, np.floating, np.integer)):
                xs.append(np.array([c]))
            else:
                return False

        self.dim = n
        self._coord_type_input = default_type
        self.x1 = self._x1_input = xs[0]
        if n >= 2:
            self.x2 = self._x2_input = xs[1]
        if n == 3:
            self.x3 = self._x3_input = xs[2]

        if self.grid and n != 2:
            logger.warning('grid == True is not allowed for dim != 2 (yet).'
                           'Turning grid = False.')
            self.grid = False

        return True

    def __iter__(self):
        if self.grid:
            raise TypeError('Grid is not iterable at the moment.')
        if self.dim == 1:
            for psi in self.psi:
                yield psi
        elif self.dim == 2:
            for i in np.arange(len(self.x1)):
                r = self.x1[i]
                z = self.x2[i]
                yield r, z

    def __len__(self):
        if self.grid:
            return len(self.x1) * len(self.x2)
        else:
            return len(self.x1)

    def __eq__(self, other):
        if isinstance(other, Coordinates):
            if self.dim != other.dim or self.grid != other.grid or len(self) != len(other):
                return False
            if self.dim == 0:
                return True
            elif self.dim == 1:
                return np.allclose(self.psi_n, other.psi_n)
            elif self.dim == 2:
                return np.allclose(self.R, other.R) and np.allclose(self.Z, other.Z)
            elif self.dim == 3:
                return np.allclose(self.R, other.R) and np.allclose(self.Z, other.Z) and np.allclose(self.phi,
                                                                                                     other.phi)
        return False

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
        if not hasattr(self, '_psi'):
            if self.dim == 1:
                self._psi = self._eq._psi_axis + self.x1 * (self._eq._psi_lcfs - self._eq._psi_axis)
            elif self.dim >= 2:
                psi = self._eq._spl_psi(self.x1, self.x2, grid=self.grid)
                if self.grid:
                    self._psi = psi.T
                else:
                    self._psi = psi
        return self._psi

    @property
    def psi_n(self):
        if not hasattr(self, '_psi_n'):
            if self.dim == 1:
                self._psi_n = self.x1
            elif self.dim >= 2:
                self._psi_n = (self.psi - self._eq._psi_axis) / (self._eq._psi_lcfs - self._eq._psi_axis)
        return self._psi_n

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
        cc_coef = - self.cocos_dict['sigma_pol'] * self.cocos_dict['sigma_cyl']
        return np.arctan2(cc_coef * (self.x2 - z_mgax), (self.x1 - r_mgax))

    @property
    def r_mid(self):
        """
        Minor radius on the outer (magnetic) midplane. Minor radius is distance from magnetic axis.

        :return: Minor radius mapped on the outer midplane.
        """
        return self._eq._rmid_spl(self.psi)

    @property
    def R_mid(self):
        """
        Major radius on the outer (magnetic) midplane. Major radius is distance from the tokamak axis.

        :return: Major radius mapped on the outer midplane.
        """
        return self._eq._mg_axis[0] + self.r_mid

    @property
    def phi(self):
        """
        Toroidal angle.

        :return: Toroidal angle.
        """
        return self.x3

    @property
    def X(self):
        if self.dim >= 2:
            return self.R * np.cos(self.phi)

    @property
    def Y(self):
        if self.dim >= 2:
            cocos_coef = self.cocos_dict['sigma_cyl']
            return cocos_coef * self.R * np.sin(self.phi)

    def mesh(self):
        if self.dim != 2 or not self.grid:
            raise TypeError('mesh can be returned only for 2d grid coordinates.')
        return np.meshgrid(self.x1, self.x2)

    def _get_cum_theta(self):
        diff_theta = np.diff(self.theta)

        # disable formation
        # fmt: off
        diff_theta[np.abs(diff_theta) > np.pi] = (
                -np.sign(diff_theta[np.abs(diff_theta) > np.pi]) *
                (2 * np.pi - np.abs(diff_theta[np.abs(diff_theta) > np.pi]))
        )
        # fmt: on

        cum_theta = np.concatenate([[0], np.cumsum(diff_theta)]) + self.theta[0]
        return cum_theta

    @property
    def cum_theta(self):
        """
        Cumulative angle along the coordinate points.

        :return: array(N)
        """
        if not hasattr(self, '_cum_theta'):
            self._cum_theta = self._get_cum_theta()
        return self._cum_theta

    def resample(self, multiple=None):
        """
        Return new, resampled instance of `pleque.Coordinates`

        :param multiple: int, use multiple to multiply number of points.
        :return: pleque.Coordinates
        """
        # TODO: TEST ME (!!!!!!)

        grid = self.grid
        eq = self._eq
        if self.dim == 1:
            psi_n = self.psi_n
            len_psi_n = len(psi_n)
            psi_n = np.interp(np.arange(len_psi_n * multiple), multiple * np.arange(len_psi_n), psi_n)

            return Coordinates(eq, psi_n, grid=grid)

        elif self.dim == 2:
            rs = self.R
            zs = self.Z
            len_rs = len(rs)
            len_zs = len(zs)
            rs = np.interp(np.arange(len_rs * multiple), multiple * np.arange(len_rs), rs)
            zs = np.interp(np.arange(len_zs * multiple), multiple * np.arange(len_zs), zs)

            return Coordinates(eq, rs, zs, grid=grid)

        elif self.dim == 3:
            rs = self.R
            zs = self.Z
            phi = self.phi

            len_rs = len(rs)
            len_zs = len(zs)
            len_phi = len(phi)
            rs = np.interp(np.arange(len_rs * multiple), multiple * np.arange(len_rs), rs)
            zs = np.interp(np.arange(len_zs * multiple), multiple * np.arange(len_zs), zs)
            phi = np.interp(np.arange(phi * multiple), multiple * np.arange(len_phi), phi)

            return Coordinates(eq, rs, zs, phi, grid=grid)
        else:
            return Coordinates(eq)

    def resample2(self, npoints):
        """
        Implicit spline curve interpolation for the limiter, number of points must be specified

        :param coords: instance of coordinates object
        :param npoints: int - number of points of the result

        """

        ### TODO: deal with different coordinate systems and dimensions

        eq = self._eq

        dists = self.cum_length

        tck, u = splprep([self.R, self.Z], u=dists, k=1, s=0)
        t = np.linspace(np.amin(u), np.amax(u), npoints)
        rs, zs = splev(t, tck)
        new_coords = Coordinates(eq, rs, zs)

        return new_coords

    def as_RZ_mid(self):
        """Transforms 1D coordinates to 2D coordinates on the midplane

        uses r_mid and the magnetic axis equilibrium
        """
        _r_mgax, z_mgax = self._eq._mg_axis
        R = self.R_mid
        Z = np.full_like(R, z_mgax)
        coords = Coordinates(self._eq, R, Z)
        return coords

    def plot(self, ax=None, **kwargs):
        """

        :param ax: Axis to which will be plotted. Default is plt.gca()
        :param kwargs: Arguments forwarded to matplotlib plot function.
        :return:
        """
        # todo: THis function should be somewhere else. A function taking coordinates as input....
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if self.dim == 1:
            ax.plot(self.psi_n, **kwargs)
        else:
            ax.plot(self.R, self.Z, **kwargs)

    def intersection(self, coords2, dim=2) -> Union["Coordinates", None]:
        """
        input: 2 sets of coordinates
        crossection of two lines (2 sets of coordinates)

        :param dim: reduce number of dimension in which is the intersection searched
        :return: Coordinates object with intersection points or None if no intersection.
        """
        from shapely import geometry

        dim_ = np.max((dim, self.dim, coords2.dim))

        if self.grid:
            raise ValueError("grid ")
        coor1 = geometry.linestring.LineString(self.as_array(dim=dim_))
        coor2 = geometry.linestring.LineString(coords2.as_array(dim=dim_))
        intersec = coor1.intersection(coor2)
        if isinstance(intersec, geometry.MultiLineString) or intersec.is_empty:
            return None
        elif isinstance(intersec, geometry.MultiPoint):
            intersec = np.asarray([np.asarray(geom.coords).squeeze() for geom in intersec.geoms]).T
        elif isinstance(intersec, geometry.Point):
            intersec = np.atleast_2d(intersec.coords).T
        else:
            return None
        return self._eq.coordinates(R=intersec[0], Z=intersec[1], coord_type=["R", "Z"])

    def as_array(self, dim=None, coord_type=None):
        """
        Return coordinates as point rows with shape ``(N, dim)``.

        For grid coordinates the mesh is flattened in ``np.meshgrid(R, Z)``
        order, so each row is one ``(R, Z)`` point.

        :param dim: reduce the number of dimensions to dim (todo)
        :param coord_type: not effected at the moment (TODO)
        :return:
        """
        # TODO integrate with numpy _as_array

        if self.dim == 0:
            return np.array(())
        # coord_type_ = self._verify_coord_type(coord_type)
        elif dim == 1 or self.dim == 1:
            return np.asanyarray(self.x1)
        elif dim == 2 or self.dim == 2:
            if self.grid:
                x1, x2 = self.mesh()
                return np.vstack((x1.ravel(), x2.ravel())).T
                # x1 = x1.ravel()
                # x2 = x2.ravel()
                # return np.array([x1, x2]).T
            else:
                return np.atleast_2d([self.x1, self.x2]).T
        elif dim == 3 or self.dim == 3:
            # todo: replace this by split method
            return np.asarray([self.x1, self.x2, self.x3]).T

    def normal_vector(self):
        """
        Calculate limiter normal vector with fw input directly from eq class

        :param first_wall: interpolated first wall
        :return: array (3, N_vecs) of limiter elements normals of the same
        """

        ### TODO: deal with different coordinate systems and dimensions

        # There will be used first order derivation in the edges and second order derivative elsewhere
        dR = -np.diff(self.R)
        dR = np.hstack((dR, [dR[-1]]))
        dR[1:-1] = dR[:-2] + dR[1:-1]

        dZ = -np.diff(self.Z)
        dZ = np.hstack((dZ, [dZ[-1]]))
        dZ[1:-1] = dZ[:-2] + dZ[1:-1]

        lim_vec = np.vstack((dR, dZ, np.zeros(np.shape(dR))))

        pol = lim_vec / np.linalg.norm(lim_vec, axis=0)

        tor = [0, 0, 1]

        normal = np.cross(pol, tor, axis=0)
        normal = normal / np.linalg.norm(normal, axis=0)

        return normal

    def incidence_angle_sin(self, vecs):
        """

        :param vecs: array (3, N_vecs)
        :return: array of sines of angles of incidence
        """

        return flux_expansion.incidence_angle_sin(self, vecs)

    def impact_angle_sin(self):
        """
        Impact angle calculation - dot product of PFC norm and local magnetic field direction.
        Internally uses `incidence_angle_sin` function where `vecs` are replaced by the vector
        of the magnetic field.

        :return: array of impact angles sines

        """

        return flux_expansion.impact_angle_sin(self)

    def impact_angle_sin_pol_projection(self):
        """
        Impact angle calculation - dot product of PFC norm and local magnetic field direction
        poloidal projection only.
        Internally uses `incidence_angle_sin` function where `vecs` are replaced by the vector
        of the poloidal magnetic field (Bphi = 0).

        :return: array of impact angles cosines
        """

        return flux_expansion.impact_angle_cos_pol_projection(self)

    @property
    def dists(self):
        """
        distances between spatial steps along the tracked field line

        Distance is returned in psi_n for dim = 1. In meters otherwise.

        :return: self._dists
        """
        if self.grid:
            raise TypeError('The grid is used - no distances between spatial steps will be calculated')
        if not hasattr(self, '_dists'):
            if self.dim == 1:
                self._dists = (self.x1[1:] - self.x1[:-1])
            elif self.dim == 2:
                self._dists = np.sqrt((self.x1[1:] - self.x1[:-1]) ** 2 + (self.x2[1:] - self.x2[:-1]) ** 2)
            elif self.dim == 3:
                self._dists = np.sqrt((self.X[1:] - self.X[:-1]) ** 2 +
                                      (self.Y[1:] - self.Y[:-1]) ** 2 +
                                      (self.Z[1:] - self.Z[:-1]) ** 2)
        return self._dists

    @property
    def cum_length(self):
        """
        Cumulative length along the coordinate points.

        :return: array(N)
        """
        if not hasattr(self, '_cum_length'):
            self._cum_length = np.hstack((0, np.cumsum(self.dists)))
        return self._cum_length

    @property
    def length(self):
        """
        Total length along the coordinate points.

        :return: length in meters
        """
        if not hasattr(self, '_cum_length'):
            self._cum_length = np.hstack((0, np.cumsum(self.dists)))
        return self._cum_length[-1]

    def _evaluate_input(self, *coordinates, coord_type=None, **coords):
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

            coord_type_ = ()
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
                # todo: make function and use for all
                valid = list(self._valid_coordinates_3d)

                # find the index of the valid coordinate system with known coordinate order
                ii = [set(item) for item in valid].index(set(xy_name))

                actual = valid[ii]
                self._x1_input = np.atleast_1d(coords[actual[0]])
                self._x2_input = np.atleast_1d(coords[actual[1]])
                self._x3_input = np.atleast_1d(coords[actual[2]])
                coord_type_ = tuple(actual)
            else:
                # self._incompatible_dimension_error(self.dim)
                raise ValueError(f'Operation in {self.dim} space has not be en allowed yet. Sorry.'
                                 )

            self._coord_type_input = coord_type_
        else:
            if len(coordinates) == 1:
                xy = coordinates[0]

                if self.grid:
                    logger.warning('grid == True is not allowed for this coordinates input. '
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

                # assume _x1_input, _x2_input and _x3_input to be arrays of size (N)
                self._x1_input = np.atleast_1d(coordinates[0])
                self._x2_input = np.atleast_1d(coordinates[1])
                self._x3_input = np.atleast_1d(coordinates[2])

            else:
                self._incompatible_dimension_error(len(coordinates))

            self._coord_type_input = self._verify_coord_type(coord_type)

        self._convert_to_default_coord_type()

        if self.dim != 2 and self.grid:
            logger.warning('grid == True is not allowed for dim != 2 (yet).'
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
                logger.warning("_coord_type_input is not correct. %s is not allowed. "
                               "Force set _coord_type_input = ('psi_n',)", tuple(coord_type))
        elif self.dim == 2:
            if coord_type is None:
                ret_coord_type = ('R', 'Z')
            elif tuple(coord_type) in self._valid_coordinates_2d:
                ret_coord_type = tuple(coord_type)
            elif tuple(coord_type[::-1]) in self._valid_coordinates_2d:
                ret_coord_type = tuple(coord_type[::-1])
            else:
                ret_coord_type = ('R', 'Z')
                logger.warning("_coord_type_input is not correct. %s is not allowed. "
                               "Force set _coord_type_input = ('R', 'Z')", tuple(coord_type))
        elif self.dim == 3:
            if coord_type is None:
                ret_coord_type = ('R', 'Z', 'phi')
            elif tuple(coord_type) in self._valid_coordinates_3d:
                ret_coord_type = tuple(coord_type)
            else:
                ret_coord_type = ('R', 'Z', 'phi')
                logger.warning("_coord_type_input is not correct. %s is not allowed. "
                               "Force set _coord_type_input = ('R', 'Z', 'phi')", tuple(coord_type))

        else:
            raise ValueError(f'Operation in {self.dim} space has not be en allowed yet. Sorry.'
                             )
        # todo: make order dependent!
        return ret_coord_type

    def _incompatible_dimension_error(self, dim):
        raise ValueError(f'Operation in {dim} space has not be en allowed yet. Sorry.'
                         )

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
            self.x1 = np.atleast_1d(self.x1)

        elif self.dim == 2:
            # only (R, Z) coordinates are implemented now
            if self._coord_type_input == ('R', 'Z'):
                self.x1 = self._x1_input
                self.x2 = self._x2_input
            elif self._coord_type_input == ('r', 'theta'):
                # todo COCOS
                r_mgax, z_mgax = self._eq._mg_axis
                cc = - self.cocos_dict['sigma_pol'] * self.cocos_dict['sigma_cyl']
                self.x1 = r_mgax + self._x1_input * np.cos(self._x2_input)
                self.x2 = z_mgax + cc * self._x1_input * np.sin(self._x2_input)
            self.x1 = np.atleast_1d(self.x1)
            self.x2 = np.atleast_1d(self.x2)

        elif self.dim == 3:
            # only (R, Z) coordinates are implemented now
            if set(self._coord_type_input) == {'R', 'Z', 'phi'}:
                self.x1 = self._x1_input
                self.x2 = self._x2_input
                self.x3 = self._x3_input
            elif set(self._coord_type_input) == {'X', 'Y', 'Z'}:
                # todo: COCOS
                # R(1)**2 = X(1)**2 + Y(2)**2
                # Z(2) = Z(3)
                # phi(3) = atan2(Y(2), X(1)]
                cc = self.cocos_dict['sigma_cyl']
                self.x1 = np.sqrt(self._x1_input ** 2 + self._x2_input ** 2)
                self.x2 = self._x3_input
                self.x3 = np.arctan2(cc * self._x2_input, self._x1_input)

            self.x1 = np.atleast_1d(self.x1)
            self.x2 = np.atleast_1d(self.x2)
            self.x3 = np.atleast_1d(self.x3)

    @deprecated('This function needs to be tested.')
    def line_integral(self, func, method='sum'):
        """
        func = /oint F(x,y) dl
        :param func: self - func(X, Y), Union[ndarray, int, float] or function values or 2D spline
        :param method: str, ['sum', 'trapezoid', 'simps']
        :return:
        """
        import inspect

        import numpy as np
        try:
            from scipy.integrate import simpson, trapezoid
        except ModuleNotFoundError:
            from scipy.integrate import simps as simpson
            from scipy.integrate import trapz as trapezoid

        #
        dx = np.hstack((0, np.cumsum(self.dists)))
        # first evaluate the dimension of coord - self and the function
        if self.grid:
            raise TypeError(
                'The grid is used - currently not possible to calculated the line average value from grid')

        if self.dim == 1:
            if method == 'sum':
                x1 = (self.x1[1:] - self.x1[:-1]) / 2
            else:
                x1 = self.x1

            if inspect.isclass(func) or inspect.isfunction(func):
                func_val = func(x1)
            elif isinstance(func, float) or isinstance(func, int):
                func_val = func
            elif inspect.ismodule(inspect.getmodule(func)):
                func_val = func(x1)
            else:
                if method == 'sum':
                    func_val = (func[1:] + func[:-1]) / 2
                else:
                    func_val = func

            if method == 'sum':
                line_integral = np.sum(func_val * self.dists)
            elif method in ['trapz', 'trapezoid']:
                line_integral = trapezoid(func_val, dx)
            elif method == 'simps':
                line_integral = simpson(func_val, dx)
            else:
                line_integral = None

        elif self.dim == 2:
            if method == 'sum':
                x1 = (self.x1[1:] + self.x1[:-1]) / 2
                x2 = (self.x2[1:] + self.x2[:-1]) / 2
            else:
                x1 = self.x1
                x2 = self.x2
            if inspect.isclass(func) or inspect.isfunction(func):
                func_val = func(x1, x2)
            elif isinstance(func, float) or isinstance(func, int):
                func_val = func
            elif inspect.ismodule(inspect.getmodule(func)):
                func_val = func(x1, x2)
            else:
                if method == 'sum':
                    if func.ndim == 1:
                        func_val = (func[1:] + func[:-1]) / 2
                    else:
                        func_val = (func[1:, 1:] + func[:-1, :-1]) / 2
                else:
                    func_val = func

            if method == 'sum':
                line_integral = np.sum(func_val * self.dists)
            elif method in ['trapz', 'trapezoid']:
                if func_val.ndim == 1:
                    line_integral = trapezoid(func_val, dx)
                else:
                    line_integral = trapezoid(trapezoid(func_val, x1), x2)
            elif method == 'simps':
                if func_val.ndim == 1:
                    line_integral = simpson(func_val, dx)
                else:
                    line_integral = simpson(simpson(func_val, x1), x2)
            else:
                line_integral = None

        elif self.dim == 3:
            raise TypeError('The 3D function was given - line averaged value needs 2D')

        return line_integral

    # def line_average(self, func, method="sum"):
