from shapely import geometry
import numpy as np

class FluxSurface:
    def __init__(self, coords):
        """
        Calculates geometrical properties of the flux surface. To make the conrour colsed, the first and last points in
        the passed coordinates have to be the same
        :param coords: vector (N,2), where N is the number of points in the contour of the flux surface and dim1 are the
        r, z coordinates of the contour points.
        """


        # closed surface has to have identical first and last points and then the shape is polygon
        # opened surface is linestring
        if coords[0, 0] == coords[-1, 0] and coords[0, 0] == coords[-1, 0]:
            self.__poly = geometry.polygon.Polygon(coords)
            self.__string = geometry.linestring.LineString(coords)
            self.__closed = True
        else:
            self.__string = geometry.linestring.LineString(coords)
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
        return self.__string.length * 2 * np.pi * self.centroid[0]

    @property
    def centroid(self):
        return np.array(self.__string.centroid.coords)[0]

    @property
    def volume(self):
        """
        Volume of the closed fluxsurface calculated from the area
         using Pappus centroid theorem : https://en.wikipedia.org/wiki/Pappus%27s_centroid_theorem
        :return: float
        """
        if self.__closed:
            return self.__poly.area * 2 * np.pi * self.centroid[0]
        else:
            raise Exception("Opened Flux Surface does not have area")

    @property
    def contour(self):
        """
        Fluxsurface contour points
        :return: numpy ndarray
        """
        return np.array(self.shape.exterior.coords)

    def contains(self, coords):
        if self.__closed:
            pnt = geometry.point.Point(coords)
            return self.__poly.contains(pnt)
        else:
            raise Exception("Opened Flux Surface does not have area")