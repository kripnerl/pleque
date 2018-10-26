import numpy as np
import numbers
import copy

class Affine():

    def __init__(self):
        self.name = None
        self.dim = None
        self.matrix = np.eye(3)

    def __matmul__(self, other):
        pass

    def __mul__(self, other):
        """should do dot product if other is a number, if it is a numpy array then matrix multiplication, if it is
        an affine transform it should multiply by other.matrix"""

        if isinstance(other, numbers.Number):
            ret = copy.deepcopy(self)
            ret.matrix = np.matmul()
            pass #scalar multiplication should be scaling the affine transform only
        if isinstance(other, np.ndarray):
            pass #multiplication by matrix should be returning updated tranformation depending on dim of other
        else:
            ValueError("Can be multiplited by an array or number only")

    def __rmul__(self, other):
        pass

    def __imatmul__(self, other):
        pass

    def __rmatmul__(self, other):
        pass

    #def __

    @property
    def liner_transformation:
        pass

    @property
    def translation:
        pass

