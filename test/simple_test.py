import os
import sys

import numpy as np
import xarray as xa

from pleque.core import Equilibrium
from pleque.io.test_gfile_loader import read_gfile as test_read_gfile

modpath = os.path.expanduser("/compass/home/kripner/Projects/pyTokamak.git")
if not modpath in sys.path:  # not to stack same paths continuously if it is already there
    sys.path.insert(0, modpath)


def test_qprofiles(g_file: str, eq: Equilibrium):
    from tokamak.formats import geqdsk
    import matplotlib.pyplot as plt

    eq_gfile = geqdsk.read(g_file)

    qpsi = eq_gfile['qpsi']
    psi_N = np.linspace(0, 1, len(qpsi))

    print(eq_gfile.keys())

    psin_axis = np.linspace(0, 1, 100)
    r = np.linspace(eq.r_min, eq.r_max, 100)
    z = np.linspace(eq.z_min, eq.z_max, 120)
    psi = eq.psi(R=r, Z=z)

    plt.figure()
    plt.subplot(121)
    ax = plt.gca()

    ax.contour(r, z, psi.T, 30)
    ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    ax.plot(eq._mg_axis[0], eq._mg_axis[1], 'o', color='b')
    ax.plot(eq._x_point[0], eq._x_point[1], 'x', color='r')
    ax.plot(eq._x_point2[0], eq._x_point2[1], 'x', color='r')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')
    ax.set_title(r'$\psi$')

    psi_mod = eq.psi(psi_N=psin_axis)
    tor_flux = eq.tor_flux(psi_N=psin_axis)
    q_as_grad = np.gradient(tor_flux, psi_mod)

    plt.subplot(222)
    ax = plt.gca()
    ax.plot(psi_N, qpsi, 'x', label='g-file data')
    ax.plot(psin_axis, q_as_grad, '-',
            label=r'$\mathrm{d} \Phi/\mathrm{d} \psi$')
    ax.plot(psin_axis, q_as_grad, '--', label='Equilibrium data')

    ax.legend()
    ax.set_xlabel(r'$\psi_\mathrm{N}$')
    ax.set_ylabel(r'$q$')

    plt.subplot(224)
    ax = plt.gca()
    ax.plot(psi_mod, tor_flux, label='Toroidal flux')
    ax.set_xlabel(r'$\psi$')
    ax.set_ylabel(r'$\Phi$')




def load_gfile(g_file):
    from tokamak.formats import geqdsk
    eq_gfile = geqdsk.read(g_file)

    psi = eq_gfile['psi']
    r = eq_gfile['r'][:, 0]
    z = eq_gfile['z'][0, :]

    pressure = eq_gfile['pressure']
    fpol = eq_gfile['fpol']
    psi_N = np.linspace(0, 1, len(fpol))

    eq_ds = xa.Dataset({'psi': (['Z', 'R'], psi),
                        'pressure': ('psi_N', pressure),
                        'fpol': ('psi_N', fpol)},
                       coords={'R': r,
                               'Z': z,
                               'psi_N': psi_N})

    print(eq_ds)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.contour(r, z, psi)
        plt.gca().set_aspect('equal')

    return eq_ds


def plot_over_view_2d(eq: Equilibrium):
    import matplotlib.pyplot as plt

    r = np.linspace(eq.r_min, eq.r_max, 100)
    z = np.linspace(eq.z_min, eq.z_max, 120)

    psi = eq.psi(R=r, Z=z)

    print('R: {}'.format(r.shape))
    print('Z: {}'.format(z.shape))
    print('psi: {}'.format(psi.shape))

    plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.contour(r, z, psi.T, 30)
    plt.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    plt.plot(eq._mg_axis[0], eq._mg_axis[1], 'o', color='b')
    plt.plot(eq._x_point[0], eq._x_point[1], 'x', color='r')
    plt.plot(eq._x_point2[0], eq._x_point2[1], 'x', color='r')

    plt.title(r'$\psi$')
    plt.gca().set_aspect('equal')

    plt.subplot(132)
    plt.contour(r, z, eq.B_pol(R=r, Z=z).T, 30)
    plt.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    plt.title(r'$B_\mathrm{pol}$')
    plt.gca().set_aspect('equal')

    plt.subplot(133)
    plt.contour(r, z, eq.B_tor(R=r, Z=z).T, 30)
    plt.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    plt.title(r'$B_\mathrm{tor}$')
    plt.gca().set_aspect('equal')

    plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.pcolormesh(r, z, eq.B_R(R=r, Z=z).T)
    plt.title(r'$B_\mathrm{R}$')
    plt.gca().set_aspect('equal')

    plt.subplot(132)
    plt.pcolormesh(r, z, eq.B_Z(R=r, Z=z).T)
    plt.title(r'$B_\mathrm{Z}$')
    plt.gca().set_aspect('equal')

    plt.subplot(133)
    plt.pcolormesh(r, z, np.sqrt(eq.B_R(R=r, Z=z) ** 2 + eq.B_Z(R=r, Z=z) ** 2).T)
    plt.title(r'$B_\mathrm{pol}$')
    plt.gca().set_aspect('equal')

    psi_n = np.linspace(0, 1, 100)

    plt.figure()
    plt.subplot(211)
    ax = plt.gca()
    ax.plot(psi_n, eq.pressure(psi_N=psi_n), 'C1')
    ax.set_xlabel(r'$\psi_\mathrm{N}$')
    ax.set_ylabel(r'$p [Pa]$', color='C1')

    ax2 = ax.twinx()
    ax2.plot(psi_n, eq.pprime(psi_N=psi_n), color='C2')
    ax2.set_ylabel(r"$p'$", color='C2')

    plt.subplot(212)
    ax = plt.gca()
    ax.plot(psi_n, eq.fpol(psi_N=psi_n), 'C1')
    ax.set_xlabel(r'$\psi_\mathrm{N}$')
    ax.set_ylabel(r'$f$ ', color='C1')

    ax2 = ax.twinx()
    ax2.plot(psi_n, eq.ffprime(psi_N=psi_n), 'C2')
    ax2.set_ylabel(r"$ff'$ ", color='C2')


def main():
    import numpy as np
    import matplotlib.pyplot as plt

    r = np.linspace(0.5, 2.5, 100)
    z = np.linspace(-1.5, 1.5, 200)

    r_grid, z_grid = np.meshgrid(r, z)
    el_r = 1 / 2
    el_z = 1 / 1

    def foofunc(r, z, par_r, par_z): return np.exp(-(r / par_r) ** 2 - (z / par_z) ** 2)

    # simple circle equilibrium
    psi_circle = foofunc(r_grid - 1.5, z_grid, el_r, el_z)
    # test x-point equilibrium done from 'three hills':
    psi_xpoint = 0.5 * foofunc(r_grid - 1.5, z_grid - 1.5, el_r, el_z * .5) + \
                 foofunc(r_grid - 1.5, z_grid + 1.5, el_r, el_z * .5) + \
                 psi_circle

    # psi = psi_circle.copy()
    psi = psi_xpoint.copy()

    # eq_ds = Dataset({'psi': (['Z', 'R'], psi)},
    #                 coords={'R': r,
    #                         'Z': z})

    # eq_ds = load_gfile('/compass/home/kripner/COMPU/fiesta/natural_divertor_v666.gfile')
    gfile = '/compass/Shared/Exchange/imrisek/MATLAB/COMPASS_U/Scenarios/scenario_1_baseline_eqdsk'
    eq_ds = load_gfile(gfile)

    # plt.pcolormesh(r_grid, z_grid, psi)
    # plt.gca().set_aspect('equal')
    # plt.colorbar()
    # plt.show()

    print(eq_ds)

    # eq = Equilibrium(eq_ds)

    eq = test_read_gfile(gfile)

    plot_over_view_2d(eq)

    test_qprofiles(gfile, eq)

    print(eq.fluxfuncs.fpol)

    print(eq.fluxfuncs.__dict__)

    plt.show()



if __name__ == '__main__':
    main()
