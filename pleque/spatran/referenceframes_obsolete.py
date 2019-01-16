import os
import sys
import numpy as np
from copy import copy
import copy
from sympy import geometry

modpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not modpath in sys.path:
    sys.path.insert(0, modpath)
from Affine import affine

#TODO: Class for a frame of reference

#TODO: Hierarchy of reference frames, highest should be world (tokamak), all others should have affine transformation to tokamak

#TODO: A Frame of reference should have an affine transform to its parent, should parent know about its children?

#TODO: place frames of reference for ports, specific directions of X, Y, Z axes:
# Z axis as a port axis,
# X axis in the XY plane of the tokamak
# Y given by Z and X.....
# zero vector in the position where the port axis intersects the plane defined by port flange

class World():
    def __init__(self):
        self._world_transf = affine.Transformation(affine.Identity())
        self.children = []

class ReferenceFrame():
    def __init__(self,name, parent, transformation):
        self.name = name

        if type(parent) in [ReferenceFrame, World, Port]:
            self.parent = parent
        else:
            raise TypeError("parrent has to be ReferenceFrame or Port, but is {0}".format(type(parent)))

        #self._parent_transf has to be of type affine.Transformation
        if type(transformation) is affine.Transformation:
            self._parent_transf = transformation #transformation to parent frame
        elif transformation.__class__.__base__ is affine.Affine:
            self._parent_transf = affine.Transformation(transforms = transformation)
        else:
            raise TypeError("transformation has to be of type {0}".format(type(affine.Transformation)))

        self._world_transf = copy.deepcopy(parent._world_transf) #transformation to world frame from parent
        self._world_transf.add(transformation, append=False) #transformation ro worlf frame from self
        self.children = []
        parent.children.append(self)

    def toparent(self, vector):
        return self._parent_transf(vector)

    def fromparent(self, vector):
        return self._parent_transf(vector, inverse = True)

    def toworld(self, vector):
        return self._world_transf(vector)

    def fromworld(self, vector):
        return self._world_transf(vector, inverse = True)

class Port():
    def __init__(self,name, world, radius, torangle, elevation, Rtang, direction, angle):

        self.tokamak = world
        self.Rtang = Rtang
        self.direction = direction
        self.angle = angle
        self.elevation = elevation
        self.torangle = torangle
        self.radius = radius
        self.name = name
        self._world_transf = None

        self.children = []
        self._getTransform()

    def _getTransform(self):

        # calculating the direction angle of the port in the midplane plane is needed for construction of the affine
        # transform for the port
        if self.Rtang > 0:
            position = geometry.Point2D(self.radius * np.cos(self.torangle),
                                        self.radius * np.sin(self.torangle))  # position of the port
            circle_tang = geometry.ellipse.Circle(geometry.Point2D(0, 0), self.Rtang)  # tangency circle of the port
            tangents = circle_tang.tangent_lines(position)  # tangent lines from port position to circle port

            # now it is needed to decide which tangency line is the correct one to match the direction to calculate correct
            # affine transform for the port

            # getting line pointing into tokamak and calculating rotation to get direction of the line
            if tangents[
                0].p1 == position:  # tangents contain tangnent lines specified by two points, one of it is position
                s1 = tangents[0].p1.x * tangents[0].p2.y - tangents[0].p2.x * tangents[0].p1.y
            else:
                s1 = tangents[0].p2.x * tangents[0].p1.y - tangents[0].p1.x * tangents[0].p2.y

            if self.direction == -1:  # clockwise
                if float(s1) < 0:
                    pick = 0
                else:
                    pick = 1
            elif self.direction == 1:  # counte clockwise
                if float(s1) < 0:
                    pick = 1
                else:
                    pick = 0

            if tangents[pick].p1 == position:  # calculate direction vector of the port axis
                direction = tangents[pick].p2 - tangents[pick].p1
            else:
                direction = tangents[pick].p1 - tangents[pick].p2
        else:
            direction = geometry.Point2D(-1 * self.radius * np.cos(self.torangle), -1 * self.radius * np.sin(self.torangle))

        # direction angle of the port axis in midplane plane
        direction_angle = float(np.arctan2(float(direction.y), float(direction.x)))

        ##direction of the port should be:
        # dimension 2 (Z-dimension) as the port axis
        # dimension 0 (X-Dimencion) in a plane parallel to the midplane, pointing counter-clockwise
        # dimension 1 specified automatically

        # rotate around y axis by 90 degrees to get port Z axis into the midplane plane
        orient1 = affine.Roty(np.pi / 2)

        # rotate around x axis to get port x axis into midplane
        orient2 = affine.Rotx(np.pi / 2)

        # applying orient1 and orient 2 both port-x and port-z should be in midplane

        # get the right angle (vertical rotation in tokamak frame)
        orient3 = affine.Roty(-1 * self.angle)

        # rotate to get port-z axis into the right toroidal direction
        orient4 = affine.Rotz(direction_angle)

        # shift to the right position
        orient5 = affine.Translate(np.array([self.radius * np.cos(self.torangle),
                                             self.radius * np.sin(self.torangle),
                                             self.elevation]))

        self._world_transf = affine.Transformation([orient1, orient2, orient3, orient4, orient5])

    def toworld(self, vector):
        return self._world_transf(vector)

    def toport(self, vector):
        return self._world_transf(vector, inverse = True)
