import numpy as np

import pleque.core.cocos as coc
from pleque.io.readers import read_geqdsk


def test_cocos_dict():
    for i in range(1, 9):
        for cocos_idx in [i, 10+i]:
            cocos = coc.cocos_coefs(cocos_idx)

            for k in ['exp_Bp', 'sigma_Bp', 'sigma_cyl', 'sigma_pol', 'sign_q', 'sign_pprime']:
                assert k in cocos

            if cocos_idx >= 10:
                assert cocos["exp_Bp"] == 1
            else:
                assert cocos["exp_Bp"] == 0


            if cocos_idx in [1, 11]:
                assert cocos['sigma_Bp'] == +1
                assert cocos['sigma_cyl'] == +1
                assert cocos['sigma_pol'] == +1
                assert cocos['sign_q'] == +1
                assert cocos['sign_pprime'] == -1

            if cocos_idx in [2, 12]:
                assert cocos['sigma_Bp'] == +1
                assert cocos['sigma_cyl'] == -1
                assert cocos['sigma_pol'] == +1
                assert cocos['sign_q'] == +1
                assert cocos['sign_pprime'] == -1

            if cocos_idx in [3, 13]:
                assert cocos['sigma_Bp'] == -1
                assert cocos['sigma_cyl'] == +1
                assert cocos['sigma_pol'] == -1
                assert cocos['sign_q'] == -1
                assert cocos['sign_pprime'] == +1

            if cocos_idx in [4, 14]:
                assert cocos['sigma_Bp'] == -1
                assert cocos['sigma_cyl'] == -1
                assert cocos['sigma_pol'] == -1
                assert cocos['sign_q'] == -1
                assert cocos['sign_pprime'] == +1

            if cocos_idx in [5, 15]:
                assert cocos['sigma_Bp'] == +1
                assert cocos['sigma_cyl'] == +1
                assert cocos['sigma_pol'] == -1
                assert cocos['sign_q'] == -1
                assert cocos['sign_pprime'] == -1

            if cocos_idx in [6, 16]:
                assert cocos['sigma_Bp'] == +1
                assert cocos['sigma_cyl'] == -1
                assert cocos['sigma_pol'] == -1
                assert cocos['sign_q'] == -1
                assert cocos['sign_pprime'] == -1

            if cocos_idx in [7, 17]:
                assert cocos['sigma_Bp'] == -1
                assert cocos['sigma_cyl'] == +1
                assert cocos['sigma_pol'] == +1
                assert cocos['sign_q'] == +1
                assert cocos['sign_pprime'] == +1

            if cocos_idx in [8, 18]:
                assert cocos['sigma_Bp'] == -1
                assert cocos['sigma_cyl'] == -1
                assert cocos['sigma_pol'] == +1
                assert cocos['sign_q'] == +1
                assert cocos['sign_pprime'] == +1


def test_coordinates_transforms(geqdsk_file):
    """
    Octants: https://en.wikipedia.org/wiki/Octant_(solid_geometry)#/media/File:Octant_numbers.svg

    :param geqdsk_file:
    :return:
    """

    for cocos in range(1, 9):
        print('COCOS: {}'.format(cocos))

        eq = read_geqdsk(geqdsk_file, cocos=cocos)
        mgax = eq.magnetic_axis

        # 3d octants:
        o1 = eq.coordinates(X=1, Y=1, Z=1)
        o2 = eq.coordinates(X=-1, Y=1, Z=1)
        o3 = eq.coordinates(X=-1, Y=-1, Z=1)
        o4 = eq.coordinates(X=1, Y=-1, Z=1)

        o5 = eq.coordinates(X=1, Y=1, Z=-1)
        o6 = eq.coordinates(X=-1, Y=1, Z=-1)
        o7 = eq.coordinates(X=-1, Y=-1, Z=-1)
        o8 = eq.coordinates(X=1, Y=-1, Z=-1)

        # 2d quadrants:
        q1 = eq.coordinates(R=mgax.R + 0.5, Z=mgax.Z + 0.5)
        q2 = eq.coordinates(R=mgax.R - 0.5, Z=mgax.Z + 0.5)
        q3 = eq.coordinates(R=mgax.R - 0.5, Z=mgax.Z - 0.5)
        q4 = eq.coordinates(R=mgax.R + 0.5, Z=mgax.Z - 0.5)

        # toroidal angle:
        if cocos in [1, 3, 5, 7]:
            # Cnt-clockwise
            assert 0 < np.mod(o1.phi, 2 * np.pi) < np.pi / 2
            assert np.pi / 2 < np.mod(o2.phi, 2 * np.pi) < np.pi
            assert np.pi < np.mod(o3.phi, 2 * np.pi) < 3 * np.pi / 2
            assert 3 * np.pi / 2 < np.mod(o4.phi, 2 * np.pi) < 2 * np.pi

            assert 0 < np.mod(o5.phi, 2 * np.pi) < np.pi / 2
            assert np.pi / 2 < np.mod(o6.phi, 2 * np.pi) < np.pi
            assert np.pi < np.mod(o7.phi, 2 * np.pi) < 3 * np.pi / 2
            assert 3 * np.pi / 2 < np.mod(o8.phi, 2 * np.pi) < 2 * np.pi

        else:
            # Clockwise
            assert 0 < np.mod(o4.phi, 2 * np.pi) < np.pi / 2
            assert np.pi / 2 < np.mod(o3.phi, 2 * np.pi) < np.pi
            assert np.pi < np.mod(o2.phi, 2 * np.pi) < 3 * np.pi / 2
            assert 3 * np.pi / 2 < np.mod(o1.phi, 2 * np.pi) < 2 * np.pi

            assert 0 < np.mod(o8.phi, 2 * np.pi) < np.pi / 2
            assert np.pi / 2 < np.mod(o7.phi, 2 * np.pi) < np.pi
            assert np.pi < np.mod(o6.phi, 2 * np.pi) < 3 * np.pi / 2
            assert 3 * np.pi / 2 < np.mod(o5.phi, 2 * np.pi) < 2 * np.pi

        # Poloidal angle:
        if cocos in [2, 3, 5, 8]:
            # Cnt-clockwise
            assert 0 * np.pi < np.mod(q1.theta[0], 2 * np.pi) < np.pi / 2
            assert np.pi / 2 < np.mod(q2.theta[0], 2 * np.pi) < np.pi
            assert np.pi < np.mod(q3.theta[0], 2 * np.pi) < 3 * np.pi / 2
            assert 3 * np.pi / 2 < np.mod(q4.theta[0], 2 * np.pi) < 2 * np.pi
        else:
            assert 0 * np.pi < np.mod(q4.theta[0], 2 * np.pi) < np.pi / 2
            assert np.pi / 2 < np.mod(q3.theta[0], 2 * np.pi) < np.pi
            assert np.pi < np.mod(q2.theta[0], 2 * np.pi) < 3 * np.pi / 2
            assert 3 * np.pi / 2 < np.mod(q1.theta[0], 2 * np.pi) < 2 * np.pi

        # Toroidal quadrants::
        o1 = eq.coordinates(R=mgax.R, Z=mgax.Z, phi=np.pi / 4)
        o2 = eq.coordinates(R=mgax.R, Z=mgax.Z, phi=3 * np.pi / 4)
        o3 = eq.coordinates(R=mgax.R, Z=mgax.Z, phi=5 * np.pi / 4)
        o4 = eq.coordinates(R=mgax.R, Z=mgax.Z, phi=7 * np.pi / 4)

        # Poloidal quadrants:
        q1 = eq.coordinates(r=0.1, theta=1 * np.pi / 4)
        q2 = eq.coordinates(r=0.1, theta=3 * np.pi / 4)
        q3 = eq.coordinates(r=0.1, theta=5 * np.pi / 4)
        q4 = eq.coordinates(r=0.1, theta=7 * np.pi / 4)

        # toroidal angle:
        if cocos in [1, 3, 5, 7]:
            # Cnt-clockwise
            assert 0 < o1.X
            assert 0 < o1.Y

            assert 0 > o2.X
            assert 0 < o2.Y

            assert 0 > o3.X
            assert 0 > o3.Y

            assert 0 < o4.X
            assert 0 > o4.Y
        else:
            # Clockwise
            assert 0 < o1.X
            assert 0 > o1.Y

            assert 0 > o2.X
            assert 0 > o2.Y

            assert 0 > o3.X
            assert 0 < o3.Y

            assert 0 < o4.X
            assert 0 < o4.Y

        # Poloidal angle:
        if cocos in [2, 3, 5, 8]:
            # Cnt-clockwise
            assert mgax.R < q1.R
            assert mgax.Z < q1.Z

            assert mgax.R > q2.R
            assert mgax.Z < q2.Z

            assert mgax.R > q3.R
            assert mgax.Z > q3.Z

            assert mgax.R < q4.R
            assert mgax.Z > q4.Z
        else:
            # Clockwise
            assert mgax.R < q1.R
            assert mgax.Z > q1.Z

            assert mgax.R > q2.R
            assert mgax.Z > q2.Z

            assert mgax.R > q3.R
            assert mgax.Z < q3.Z

            assert mgax.R < q4.R
            assert mgax.Z < q4.Z
