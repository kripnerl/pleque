import numpy as np


#TODO: Now it made to tranform single vectors, we should implement also the possibility to transform arrays of vectors

class Affine():

    def __init__(self):
        self.transformation = None
        self.inverse = None
        self.type = "transformation"

    def __call__(self, vector, inverse=False):
        """
        Calling the class performs the transformation
        :param vector: Vector to be transformed
        :param inverse: Should the iverse transform be used?
        :return: transformed vector
        """
        if inverse:
            transformation = self.inverse
        else:
            transformation = self.transformation

        #check of input vector shape and form
        if vector.shape[0] < 3 and vector.shape[0] > 4:
            raise ValueError("dim 1 of vector has to be length 3 or 4")
        if np.ndim(vector) > 2:
            raise ValueError("dim 1 of vector has to be length 3 or 4")

        if np.ndim(vector) == 1:
            if vector.shape[0] == 3:
                vector = np.append(vector, [1])

            return np.matmul(transformation, vector)[0:3]

        elif np.ndim(vector) == 2:
            if vector.shape[0] == 3:
                vector = np.concatenate((vector, np.ones((1, vector.shape[1]))), axis=0)

            return np.matmul(transformation, vector)[0:3,:]


    def volumepreserved(self):
        """
        Linear tranformation preserves volume if and only if its determinant is equal to 1. Translation preserves volume
        so determinant of transformation[0:4, 0:4] is calculated,
        :return:
        """
        det = np.linalg.det(self.transformation[0:4, 0:4])
        if det == 1:
            return True
        else:
            return False

    def anglepreserved(self):
        """
        Linear tranformation preserves volume if and only if its determinant is equal to 1. Translation preserves angle
        so determinant of transformation[0:4, 0:4] is calculated,
        :return:
        """
        det = np.linalg.det(self.transformation[0:4, 0:4])
        if det == 1 or det == -1:
            return True
        else:
            return False

class Identity(Affine):

    def __init__(self):
        super()
        self.type = "identity"
        self.transformation = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
        self.inverse = self.transformation

class Translate(Affine):

    def __init__(self, transformation):
        """
        3D Affine translation class
        :param transformation: translation vector of the length 3
        """
        super()
        self.type = "translation" #name of the transofrmation
        if type(transformation) is np.ndarray:
            self.tranformation = transformation
        elif type(transformation) is list:
            self.tranformation = np.array(transformation, dtype = float)
        else:
            raise ValueError("transformation is type {0}, but should be list or numpy array".format(type(transformation)))

        if not transformation.shape[0] == 3:
            raise ValueError("Transformation has lenght {0} but has to have length 3".format(transformation.shape[0]))

        self.translation = transformation

        self.transformation = np.array([[1, 0, 0, transformation[0]],
                        [0, 1, 0, transformation[1]],
                        [0, 0, 1, transformation[2]],
                        [0, 0, 0, 1]])

        self.inverse = np.array([[1, 0, 0, -1*transformation[0]],
                        [0, 1, 0, -1*transformation[1]],
                        [0, 0, 1, -1*transformation[2]],
                        [0, 0, 0, 1]])

class Scale(Affine):

    def __init__(self, transformation):
        """
        3D Affine translation class
        :param transformation: translation vector of the length 3
        """
        super()
        self.type = "scale" #name of the transofrmation
        if type(transformation) is np.ndarray:
            self.tranformation = transformation
        elif type(transformation) is list:
            self.tranformation = np.array(transformation, dtype = float)
        else:
            raise ValueError("transformation is type {0}, but should be list or numpy array".format(type(transformation)))

        if not transformation.shape[0] == 3:
            raise ValueError("Transformation has lenght {0} but has to have length 3".format(transformation.shape[0]))

        self.scale = transformation

        self.transformation = np.array([[transformation[0], 0, 0, 0],
                        [0, transformation[1], 0, 0],
                        [0, 0, transformation[2], 0],
                        [0, 0, 0, 1]])

        self.inverse = np.array([[1./transformation[0], 0, 0, 0],
                        [0, 1./transformation[1], 0, 0],
                        [0, 0, 1./transformation[2], 0],
                        [0, 0, 0, 1]])

class Rotx(Affine):

    def __init__(self, transformation, rad = True):
        super()
        self.type = "Rotation-x" #name of the transofrmation

        if not type(transformation) in [int, float]:
            raise ValueError("Transformation should be of type float or int, but type {0} was passed".format(transformation))
        elif type(transformation) is np.ndarray:
            if transformation.shape.shape[0] > 1 or transformation.shape[0] > 1:
                raise ValueError("Only a single angle can be specified")

        if rad:
            self.radians = transformation
            self.degrees = np.rad2deg(transformation)
        else:
            self.degrees = transformation
            self.radians = np.deg2rad(transformation)

        self.transformation = np.array([[1, 0, 0, 0],
                        [0, np.cos(self.radians), -1*np.sin(self.radians), 0],
                        [0, np.sin(self.radians), np.cos(self.radians), 0],
                        [0, 0, 0, 1]])

        self.inverse = np.array([[1, 0, 0, 0],
                        [0, np.cos(-1*self.radians), -1*np.sin(-1*self.radians), 0],
                        [0, np.sin(-1*self.radians), np.cos(-1*self.radians), 0],
                        [0, 0, 0, 1]])

class Roty(Affine):

    def __init__(self, transformation, rad = True):
        super()
        self.type = "Rotation-y" #name of the transofrmation

        if not type(transformation) in [int, float]:
            raise ValueError("Transformation should be of type float or int, but type {0} was passed".format(transformation))
        elif type(transformation) is np.ndarray:
            if transformation.shape.shape[0] > 1 or transformation.shape[0] > 1:
                raise ValueError("Only a single angle can be specified")

        if rad:
            self.radians = transformation
            self.degrees = np.rad2deg(transformation)
        else:
            self.degrees = transformation
            self.radians = np.deg2rad(transformation)

        self.transformation = np.array([[np.cos(self.radians), 0, np.sin(self.radians), 0],
                        [0, 1, 0, 0],
                        [-1*np.sin(self.radians), 0, np.cos(self.radians), 0],
                        [0, 0, 0, 1]])

        self.inverse = np.array([[np.cos(-1*self.radians), 0, np.sin(-1*self.radians), 0],
                        [0, 1, 0, 0],
                        [-1*np.sin(-1*self.radians), 0, np.cos(-1*self.radians), 0],
                        [0, 0, 0, 1]])

class Rotz(Affine):

    def __init__(self, transformation, rad = True):
        super()
        self.type = "Rotation-z" #name of the transofrmation

        if not type(transformation) in [int, float]:
            raise ValueError("Transformation should be of type float or int, but type {0} was passed".format(transformation))
        elif type(transformation) is np.ndarray:
            if transformation.shape.shape[0] > 1 or transformation.shape[0] > 1:
                raise ValueError("Only a single angle can be specified")

        if rad:
            self.radians = transformation
            self.degrees = np.rad2deg(transformation)
        else:
            self.degrees = transformation
            self.radians = np.deg2rad(transformation)

        self.transformation = np.array([[np.cos(self.radians), -1 * np.sin(self.radians), 0, 0],
                        [np.sin(self.radians), np.cos(self.radians), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        self.inverse = np.array([[np.cos(-1*self.radians), -1 * np.sin(-1*self.radians), 0, 0],
                        [np.sin(-1*self.radians), np.cos(-1*self.radians), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

class Transformation(Affine):

    def __init__(self, transforms = None):
        """
        Chain of Affine transforms
        :param transforms:
        """
        super()
        self._chain = [] #list containing individual affine transforms

        self.transformation = None # transformation obtained by chaining affine transforms in chain
        self.inverse = None # inverse affine transform corresponding to the tranformation

        if transforms:
            self.add(transforms)

    def add(self,transform, append = True):
        """
        Appends list or single affine transforms to the transformation chain and calculates new forward and inverse
        transformations
        :param transform: single or list of Affine transforms objects to be added to the chain
        :return:
        """

        if type(transform) is list:
            for i in transform:
                if i.__class__.__base__ is Affine:
                    if append:
                        self._chain.append(i)
                    else:
                        self._chain.insert(0,i)
                else:
                    raise ValueError("transformations have to be of type Affine")
        elif type(transform) is Transformation: ##adding two chains should be possible
            if append:
                self._chain = self._chain + transform._chain
            else:
                self._chain =  transform._chain + self._chain
        elif transform.__class__.__base__ is Affine:
            if append:
                self._chain.append(transform)
            else:
                self._chain.insert(0,transform)
        self._calculatetransform()
        self._calculateinverse()

    def _calculatetransform(self):
        """
        recalculate transformation
        :return:
        """
        self.transformation = np.identity(4)

        for i in self._chain:
                self.transformation = np.matmul(i.transformation, self.transformation)

    def _calculateinverse(self):
        """
        recalculate inverse transformation
        :return:
        """
        self.inverse = np.identity(4)
        for i in reversed(self._chain):
                self.inverse = np.matmul(i.inverse, self.inverse)