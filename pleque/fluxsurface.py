import numpy as np
from shapely import geometry

from pleque.core import Coordinates


class FluxSurface:
    def __init__(self, coords: Coordinates):
        """
        Calculates geometrical properties of the flux surface. To make the conrour colsed, the first and last points in
        the passed coordinates have to be the same.
        Instance is obtained by calling method `flux_surface` in instance of `Equilibrium`.
        :param coords: Instance of coordinate class
        """
        self.coords = coords
        points_RZ = coords.as_array(('R', 'Z'))
        # closed surface has to have identical first and last points and then the shape is polygon
        # opened surface is linestring
        if points_RZ[0, 0] == points_RZ[-1, 0] and points_RZ[0, 1] == points_RZ[-1, 1]:
            self.__poly = geometry.polygon.Polygon(points_RZ)
            self.__string = geometry.linestring.LineString(points_RZ)
            self.__closed = True
        else:
            self.__string = geometry.linestring.LineString(points_RZ)
            self.__closed = False

    @property
    def closed(self):
        """
        True if the fluxsurface is closed.
        :return:
        """
        return self.__closed

    @property
    def area(self):
        """
        Area of the closed fluxsurface.
        :return:
        """
        if self.__closed:
            return self.__poly.area
        else:
            raise Exception("Opened Flux Surface does not have area")

    @property
    def length(self):
        """
        Length of the fluxsurface contour
        :return:
        """
        return self.__string.length

    @property
    def surface(self):
        """
        Surface of fluxsurface calculated from the contour length
         using Pappus centroid theorem : https://en.wikipedia.org/wiki/Pappus%27s_centroid_theorem
        :return: float
        """
        return self.__string.length * 2 * np.pi * self.centroid.R[0]

    @property
    def centroid(self):
        return self.coords._eq.coordinates(R=np.array(self.__string.centroid.coords)[0][0],
                                           Z=np.array(self.__string.centroid.coords)[0][0], coord_type=["R", "Z"])

    @property
    def volume(self):
        """
        Volume of the closed fluxsurface calculated from the area
         using Pappus centroid theorem : https://en.wikipedia.org/wiki/Pappus%27s_centroid_theorem
        :return: float
        """
        if self.__closed:
            return self.__poly.area * 2 * np.pi * self.centroid.R[0]
        else:
            raise Exception("Opened Flux Surface does not have area")

    @property
    def contour(self):
        """
        Fluxsurface contour points
        :return: numpy ndarray
        """
        return self.coords

    def contains(self, coords: Coordinates):
        points_RZ = coords.as_array(('R', 'Z'))[0, :]
        if self.__closed:
            pnt = geometry.point.Point(points_RZ)
            return self.__poly.contains(pnt)
        else:
            raise Exception("Opened Flux Surface does not have area")

    def distance(self, coords: Coordinates):
        point = geometry.Point(coords.as_array()[0])
        distance = self.__string.distance(point)
        return distance
