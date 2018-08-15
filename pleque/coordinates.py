import numpy as np

from pleque.core import Equilibrium


class Coordinates(object):

    def __init__(self, equilibrium: Equilibrium, *coordinates, coord_type=None, grid=False, **coords):
        self._equilibrium = equilibrium
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

    def getAs(self, type):
        pass

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

                if self.dim == 1:
                    self.x1 = xy[0]
                    self.coord_type = tuple(xy_name[0])
                if self.dim == 2:
                    if tuple(xy_name) in self._valid_coordinates_2d:
                        self.x1 = xy[0]
                        self.x2 = xy[1]
                        self.coord_type = tuple(xy_name)
                    elif tuple(xy_name[::-1]) in self._valid_coordinates_2d:
                        self.x1 = xy[1]
                        self.x2 = xy[0]
                        self.coord_type = tuple(xy_name[::-1])
                        if len(self.x1) != len(self.x2):
                            raise ValueError('All coordinates should contain same dimension.')
                    else:
                        raise ValueError('Invalid combination of input coordinates.')
                    self.coord_type = (xy_name[0],)
                if self.dim >= 3:
                    self._incompatible_dimension(self.dim)

        elif len(coordinates) == 1:
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
                    self.x1 = xy[:, 0]
                elif self.dim == 2:
                    self.x1 = xy[:, 0]
                    self.x2 = xy[:, 1]
                else:
                    self._incompatible_dimension(self.dim)
            else:
                # 1d, one number
                self.dim = 1
                self.x1 = np.array([xy])
            pass
        elif len(coordinates) == 2:
            self.dim = 2
            x1 = coordinates[0]
            x2 = coordinates[1]

            # assume x1 and x2 to be arrays of size (N)
            if not isinstance(x1, np.ndarray):
                x1 = np.array(x1, ndmin=1)
            if not isinstance(x2, np.ndarray):
                x2 = np.array(x2, ndmin=1)
            self.x1 = x1
            self.x2 = x2

        else:
            self._incompatible_dimension(len(coordinates))

        # set coordinates type:
        if isinstance(coord_type, str):
            coord_type = (coord_type,)

        if self.dim == 0:
            self.coord_type = ()
        elif self.dim == 1:
            if coord_type is None:
                self.coord_type = ('psi_n',)
            elif tuple(coord_type) in self._valid_coordinates_1d:
                self.coord_type = tuple(coord_type)
            else:
                self.coord_type = ('psi_n',)
                print("WARNING: coord_type not correctly set. \n"
                      "{} is not allowed \n"
                      "Force set coord_type = ('psi_n',)"
                      .format(tuple(coord_type)))
        elif self.dim == 2:
            if coord_type is None:
                self.coord_type = ('R', 'Z')
            elif tuple(coord_type) in self._valid_coordinates_2d:
                self.coord_type = tuple(coord_type)
            elif tuple(coord_type[::-1]) in self._valid_coordinates_2d:
                self.coord_type = tuple(coord_type[::-1])
            else:
                self.coord_type = ('R', 'Z')
                print("WARNING: coord_type not correctly set. \n"
                      "Force set coord_type = ('R', 'Z')")
        else:
            self._incompatible_dimension(self.dim)

    def _incompatible_dimension(self, dim):
        raise ValueError('Operation in {} space has not been allowed yet. Sorry.'
                         .format(dim))
