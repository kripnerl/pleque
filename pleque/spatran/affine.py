import copy

import numpy as np

from pleque.core import Coordinates


class Affine:

    def __init__(self, transform: np.ndarray = np.eye(4), name: str = ""):
        """
        Affine transformation class.

        :param name: name of the affine transformation
        :param transform: Affine transformation matrix of the snape (ndim+1, ndim+1)
        """
        self.name = name
        self.transformation = transform
        self.dim = self.transformation.shape[0] - 1
        self.transformation_inverse = np.zeros_like(transform)
        self._recalculate_inverse()

    def __mul__(self, other):
        """
        Multiplication of Affine object from left

        :param other: the multiplicator. If it is of type Affine, the Affine transformation is updated by
            multiplication from left.
            If it is of type np.ndarray of shape (dim, n), understand n column vectors to be transformed, a new
            transformed vector or vectors are returned.
        :return:
        """

        # if passed vector is (dim,n), make it 4-vector (dim+1,1)

        if isinstance(other, Affine):
            return self._calculate_new_transformation(self.transformation, other.transformation)
        elif isinstance(other, np.ndarray):
            if len(other.shape) == 1:
                other = np.atleast_2d(other).T

            if other.shape[0] == self.dim:
                other = np.concatenate((other, np.ones((1, other.shape[1]))), axis=0)
                return self._transform_vector(other)[0:-1, :]
            elif other.shape[0] == self.dim + 1:
                return self._transform_vector(other)[0:-1, :]
            else:
                raise ValueError("Array has to be of shape (dim, n) or (dim+1, n)")
        elif isinstance(other, Coordinates):
            # todo: Add treatment of coordinates objectj
            pass

    def __invert__(self):
        """
        Invert the transformation, i.e. forward Affine transform becomes reverse and vice versa
        :return: Affine transformation
        """
        backwards = self.transformation
        forward = self.transformation_inverse
        ret = copy.deepcopy(self)

        ret.transformation = forward
        ret.transformation_inverse = backwards

        return ret

    def __rmul__(self, other):
        """
        MUltiplication of Affine object from left
        :param other: the multiplicator. If it is of type Affine, the Affine transformation is updated by
            multiplication from left.
            If it is of type np.ndarray of shape (dim, n), understand n column vectors to be transformed, a new
            transformed vector or vectors are returned.
        :return:
        """
        if isinstance(other, Affine):
            self._calculate_new_transformation(self.transformation, other.transformation)
        else:
            ValueError("Affine transfrom can be multiplied from left only by an Affine transform")

    def _transform_vector(self, other):
        return np.matmul(self.transformation, other)

    def _calculate_new_transformation(self, left, right):
        """should do dot product if other is a number, if it is a numpy array then matrix multiplication, if it is
        an affine transform it should multiply by other.matrix"""

        new_transformation = np.matmul(left, right)

        ret = copy.deepcopy(self)
        ret.transformation = new_transformation
        ret._recalculate_inverse()

        return ret

    def _recalculate_inverse(self):
        linear_part = np.eye(self.dim + 1)
        linear_part[0:-1, 0:-1] = np.linalg.inv(self.transformation_linear)
        translation = -1 * np.matmul(linear_part, self.transformation_translation)
        self.transformation_inverse[0:self.dim + 1, 0:self.dim + 1] = linear_part
        self.transformation_inverse[:, -1] = translation

    def volumepreserved(self):
        """
        Does the transformation preserve volume?
        :return:
        """
        det = np.linalg.det(self.transformation)
        if det == 1:
            return True
        else:
            return False

    def anglepreserved(self):
        """
        Does the transformation preserves angle?
        :return:
        """
        det = np.linalg.det(self.transformation)
        if det == 1 or det == -1:
            return True
        else:
            return False

    @property
    def transformation_linear(self):
        """
        Linear transformation part of the affine transformation
        :return:
        """
        return self.transformation[0:self.dim, 0:self.dim]

    @property
    def transformation_translation(self):
        """
        translation vector of the affine transformation
        :return:
        """
        return self.transformation[:, -1]


class Identity(Affine):

    def __init__(self, dim: int = 3, name: str = ""):
        transform = np.eye(dim + 1)
        super().__init__(transform, name)


class Translate(Affine):

    def __init__(self, transform, name: str = ""):
        dim = transform.shape[0]
        transformation = np.eye(dim + 1)
        transformation[0:dim, -1] = transform
        super().__init__(transformation, name)


class Scale(Affine):

    def __init__(self, transform, name: str = ""):
        dim = transform.shape[0]
        transformation = np.eye(dim + 1)
        # for i in range(dim):
        #    transformation[i,i] = transform[i]
        diag = np.diag_indices_from(transformation[0:-1, 0:-1])
        transformation[diag] = transform

        super().__init__(transformation, name)


class Rotx(Affine):

    def __init__(self, angle, name: str = "", rad=True):
        if rad:
            radians = angle
        else:
            radians = np.deg2rad(angle)

        transformation = np.array([[1, 0, 0, 0],
                                   [0, np.cos(radians), -1 * np.sin(radians), 0],
                                   [0, np.sin(radians), np.cos(radians), 0],
                                   [0, 0, 0, 1]])
        super().__init__(transformation, name)


class Roty(Affine):
    def __init__(self, angle, name: str = "", rad=True):
        if rad:
            radians = angle
        else:
            radians = np.deg2rad(angle)

        transformation = np.array([[np.cos(radians), 0, np.sin(radians), 0],
                                   [0, 1, 0, 0],
                                   [-1 * np.sin(radians), 0, np.cos(radians), 0],
                                   [0, 0, 0, 1]])
        super().__init__(transformation, name)


class Rotz(Affine):

    def __init__(self, angle, name: str = "", rad=True):
        if rad:
            radians = angle
        else:
            radians = np.deg2rad(angle)

        transformation = np.array([[np.cos(radians), -1 * np.sin(radians), 0, 0],
                                   [np.sin(radians), np.cos(radians), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

        super().__init__(transformation, name)
