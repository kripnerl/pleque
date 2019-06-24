import numpy as np

import pleque.core.cocos as coc
from pleque.core import Equilibrium
from pleque.io.readers import read_geqdsk
from pleque.io._geqdsk import read as read_geqdsk_as_dict, data_as_ds

import pytest

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


@pytest.mark.parametrize(('cocos',), [[1], [2], [3], [4]])  # , [5], [6], [7], [8]])
def test_directions(geqdsk_file, cocos):
    print('g-file: {}'.format(geqdsk_file))
    print('COCOS: {}'.format(cocos))

    direction = -1

    eq = read_geqdsk(geqdsk_file, cocos=cocos)
    mgax = eq.magnetic_axis
    maxx = eq.Z_max

    R0 = mgax.R + 0.5 * (maxx - mgax.R)
    Z0 = mgax.Z - 0.06
    p0 = eq.coordinates(R=R0, Z=Z0, phi=0)

    tr = eq.trace_field_line(R=R0, Z=Z0, direction=direction)[0]

    br = eq.B_R(R=R0, Z=Z0)
    bz = eq.B_Z(R=R0, Z=Z0)
    btor = eq.B_tor(R=R0, Z=Z0)
    coef = 0.01

    p1 = eq.coordinates(R=R0 + coef * br, Z=Z0 + coef * bz, phi=coef * btor)

    dphi = tr.phi[1] - tr.phi[0]
    dtheta = tr.theta[1] - tr.theta[0]

    assert np.sign(dphi) == np.sign(p1.phi - p0.phi) * direction
    assert np.sign(dtheta) == np.sign(p1.theta - p0.theta) * direction

    assert np.isclose(tr.theta[0], tr.theta[-1], atol=0.1, rtol=0.1)

    dphidtheta = np.sign(dphi * dtheta)
    my_dir = eq._cocosdic['sigma_pol'] * np.sign(eq.I_plasma) * np.sign(eq.F0)
    # my_dir = eq._psi_sign * eq._cocosdic['sigma_pol'] * eq._cocosdic['sigma_cyl']* eq._cocosdic['sigma_pol'] * np.sign(eq.I_plasma) * np.sign(eq.F0)

    print('sigma_cyl: {}\nsigma_pol: {}\nIp: {}\nF0: {}'.format(
        eq._cocosdic['sigma_cyl'], eq._cocosdic['sigma_pol'], eq.I_plasma, eq.F0
    ))
    print('sign psi: {}'.format(eq._psi_sign))
    print('dphi = {}\ndtheta = {}'.format(dphi, dtheta))

    assert my_dir == dphidtheta


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


def test_cocos_consistency(geqdsk_file):
    for cocos in range(1, 9):
        print('COCOS: {}'.format(cocos))

        equilibrium = read_geqdsk(geqdsk_file, cocos=cocos)
        with open(geqdsk_file, 'r') as f:
            eq_dict = read_geqdsk_as_dict(f)
            eq_xr = data_as_ds(eq_dict)
            eq_xr.psi.values = eq_xr.psi * (2 * np.pi)
            fw = np.stack((eq_xr['r_lim'].values, eq_xr['z_lim'].values)).T  # first wall
            equilibrium2 = Equilibrium(eq_xr, fw, cocos=cocos + 10)

        sigma_Ip = np.sign(equilibrium.I_plasma)
        sigma_B0 = np.sign(equilibrium.F0)
        cocos_dict = equilibrium._cocosdic

        Rax = equilibrium.magnetic_axis.R[0]
        Zax = equilibrium.magnetic_axis.Z[0]

        assert sigma_B0 * sigma_Ip == np.sign(equilibrium.q(psi_n=0.5)) * cocos_dict['sigma_pol']
        assert sigma_Ip == cocos_dict['sigma_Bp'] * equilibrium._psi_sign

        assert np.sign(equilibrium.F(R=Rax, Z=Zax)) == sigma_B0
        assert np.sign(equilibrium.B_tor(R=Rax, Z=Zax)) == sigma_B0

        assert np.sign(equilibrium.j_tor(r=0.1, theta=0)) == sigma_Ip
        assert np.sign(equilibrium.psi(psi_n=1) - equilibrium.psi(psi_n=0)) == sigma_Ip * cocos_dict['sigma_Bp']
        assert np.sign(equilibrium.tor_flux(r=0.1, theta=0)) == sigma_B0

        assert np.sign(equilibrium.pprime(psi_n=0.5)) == - sigma_Ip * cocos_dict['sigma_Bp']
        assert np.sign(equilibrium.q(psi_n=0.5)) == sigma_Ip * sigma_B0 * cocos_dict['sigma_pol']

        assert np.isclose(
            equilibrium.j_tor(R=equilibrium.magnetic_axis.R + 0.5, Z=equilibrium.magnetic_axis.Z),
            equilibrium2.j_tor(R=equilibrium2.magnetic_axis.R + 0.5, Z=equilibrium2.magnetic_axis.Z))

        assert np.isclose(equilibrium.q(psi_n=0.5), equilibrium2.q(psi_n=0.5))
        assert np.isclose(
            equilibrium.I_plasma, equilibrium2.I_plasma,
            atol=100
        )

        # todo test poloidal current direction (!!!)
