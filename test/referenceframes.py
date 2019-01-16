from sympy import geometry
from pleque.spatran import affine
from pleque.spatran.reference_frames import ReferenceFrame
import numpy as np

def get_tangentpoints(r_tangency, pos_torangle, pos_rmajor):

    a = np.square(r_tangency)/pos_rmajor
    b = r_tangency * np.sqrt(1-np.square(r_tangency/pos_rmajor))

    p1 = [a * np.cos(pos_torangle) - b * np.sin(pos_torangle),
          a * np.sin(pos_torangle) + b * np.cos(pos_torangle)]

    p2 = [a * np.cos(pos_torangle) + b * np.sin(pos_torangle),
          a * np.sin(pos_torangle) - b * np.cos(pos_torangle)]

    return p1, p2



if __name__ == "__main__":
    if False:
        Rtang = 0.89
        radius = 5
        torangle = np.deg2rad(45)


        tan1, tan2 = get_tangentpoints(Rtang, torangle, radius)

    if True:
        trans = affine.Translate(np.array([1, 0, 0]))
        trans2 = affine.Translate(np.array([0, 1, 0]))
        vector = np.array([0, 0, 0])

        origin = ReferenceFrame("origin")
        child = ReferenceFrame("child", trans, origin)
        child2 = ReferenceFrame("child2", trans2, child)


        #print("child to parent {0}".format(child.toparent(vector)))
        #print("origin to child {0}".format(origin.tochild(child, vector)))
        print("child2 to child {0}".format(child2.toparent(vector)))
        print("child2 to origin {0}".format(child2.toorigin(vector)))
        print("child to child2 {0}".format(child.tochild(child, vector)))

        #print(origin.toparent(vector))