from collections.abc import Sequence
import itertools

import numpy as np
import xarray

from pleque.utils.decorators import deprecated

class Coordinates(object):

    def __init__(self, equilibrium, *coordinates, coord_type=None, grid=False, **coords):
        """

        :param equilibrium:
        :param coordinates:
        :param coord_type:
        :param grid:
        :param coords:

        Default coordinate systems
        --------------------------

        - **1D**: :math:`\psi_\mathrm{N}`,
        - **2D**: :math:`(R, Z)`,
        - **3D**: :math:`(R, Z, \phi)`.

        Accepted coordinates types
        --------------------------

        **1D - coordinates**

        +------------------------+-----------+------------------------------+
        | Coordinate             | Code      | Note                         |
        +========================+===========+==============================+
        |:math:`\psi_\mathrm{N}` | ``psi_n`` | Default 1D coordinate        |
        +------------------------+-----------+------------------------------+
        |:math:`\psi`            | ``psi``   |                              |
        +------------------------+-----------+------------------------------+
        |:math:`\rho`            | ``rho``   | :math:`\rho = \sqrt{\psi_n}` |
        +------------------------+-----------+------------------------------+

        **2D - coordintares**

        +------------------------+--------------+-------------------------------------------------+
        | Coordinate             | Code         | Note                                            |
        +========================+==============+=================================================+
        |:math:`(R, Z)`          | ``R, Z``     | Default 2D coordinate                           |
        +------------------------+--------------+-------------------------------------------------+
        |:math:`(r, \theta)`     | ``r, theta`` | Polar coordinates with respect to magnetic axis |
        +------------------------+--------------+-------------------------------------------------+

        **3D - coordinates**

        +------------------------+---------------+-------------------------------------------------+
        | Coordinate             | Code          | Note                                            |
        +========================+===============+=================================================+
        |:math:`(R, Z, \phi)`    | ``R, Z, phi`` | Default 3D coordinate                           |
        +------------------------+---------------+-------------------------------------------------+
        |:math:`(X, Y, Z)`       | ``(X, Y, Z)`` | Polar coordinates with respect to magnetic axis |
        +------------------------+---------------+-------------------------------------------------+


        """

        self._eq = equilibrium
        self._valid_coordinates = {'R', 'Z', 'psi_n', 'psi', 'rho', 'r', 'theta', 'phi', 'X', 'Y'}
        self._valid_coordinates_1d = {('psi_n',), ('psi',), ('rho',)}
        self._valid_coordinates_2d = {('R', 'Z'), ('r', 'theta')}
        self._valid_coordinates_3d = {('R', 'Z', 'phi'), ('X', 'Y', 'Z')}
        self.dim = -1  # init only
        self.grid = grid

        self.__evaluate_input__(*coordinates, coord_type=coord_type, **coords)

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

    def plot(self, ax=None, **kwargs):
        """

        :param ax: Axis to which will be plotted. Default is plt.gca()
        :param kwargs: Arguments forwarded to matplotlib plot function.
        :return:
        """
        #todo: THis function should be somewhere else. A function taking coordinates as input....
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if self.dim == 1:
            ax.plot(self.psi_n, **kwargs)
        else:
            ax.plot(self.R, self.Z, **kwargs)

    def as_array(self, dim=None, coord_type=None):
        """
        Return array of size (N, dim), where N is number of points and dim number of dimensions specified by coord_type

        :param dim: reduce the number of dimensions to dim (todo)
        :param coord_type: not effected at the moment (TODO)
        :return:
        """
        if self.dim == 0:
            return np.array(())
        # coord_type_ = self._verify_coord_type(coord_type)
        elif dim == 1 or self.dim == 1:
            return self.x1
        elif dim == 2 or self.dim == 2:
            if self.grid:
                x1, x2 = self.mesh()
                return np.vstack((x1.ravel(), x2.ravel())).T
                # x1 = x1.ravel()
                # x2 = x2.ravel()
                # return np.array([x1, x2]).T
            else:
                return np.array([self.x1, self.x2]).T
        elif dim == 3 or self.dim == 3:
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
                #if tuple(xy_name) in self._valid_coordinates_3d:
                permutations = list(itertools.permutations(xy_name))
                # if any([p in self._valid_coordinates_3d for p in permutations]):
                #
                #     # todo: implement various order of coordinates
                #     self._x1_input = xy[0]
                #     self._x2_input = xy[1]
                #     self._x3_input = xy[2]
                #     coord_type_ = tuple(xy_name)
                # todo: make function and use for all

                valid = list(self._valid_coordinates_3d)

                # find the index of the valid coordinate system with known coordinate order
                ii = [set(item) for item in valid].index(set(xy_name))

                actual = valid[ii]
                self._x1_input = coords[actual[0]]
                self._x2_input = coords[actual[1]]
                self._x3_input = coords[actual[2]]
                coord_type_ = tuple(actual)
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
            #if self._coord_type_input == ('R', 'Z', 'phi'):
            if any([p == ('R', 'Z', 'phi') for p in itertools.permutations(self._coord_type_input)]):
                self.x1 = self._x1_input
                self.x2 = self._x2_input
                self.x3 = self._x3_input
            #elif self._coord_type_input == ('X', 'Y', 'Z'):
            elif any([p == ('X', 'Y', 'Z') for p in itertools.permutations(self._coord_type_input)]):
                # todo: COCOS
                # R(1)**2 = X(1)**2 + Y(2)**2
                # Z(2) = Z(3)
                # phi(3) = atan2(Y(2), X(1)]
                self.x1 = np.sqrt(self._x1_input ** 2 + self._x2_input ** 2)
                self.x2 = self._x3_input
                self.x3 = np.arctan2(self._x2_input, self._x1_input)
