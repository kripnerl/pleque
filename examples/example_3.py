import os
import sys

import numpy as np
import xarray as xa

from pleque.core import Equilibrium
from pleque.tests.utils import load_testing_equilibrium, get_test_equilibria_filenames

modpath = os.path.expanduser("/compass/home/kripner/Projects/pyTokamak.git")
if not modpath in sys.path:  # not to stack same paths continuously if it is already there
    sys.path.insert(0, modpath)


# noinspection PyUnreachableCode
def load_gfile(g_file):
    from pleque.io._geqdsk import read
    eq_gfile = read(g_file)

    psi = eq_gfile['psi']
    r = eq_gfile['r'][:, 0]
    z = eq_gfile['z'][0, :]

    pressure = eq_gfile['pressure']
    F = eq_gfile['F']
    psi_n = np.linspace(0, 1, len(F))

    eq_ds = xa.Dataset({'psi': (['Z', 'R'], psi),
                        'pressure': ('psi_n', pressure),
                        'F': ('psi_n', F)},
                       coords={'R': r,
                               'Z': z,
                               'psi_n': psi_n})

    print(eq_ds)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.contour(r, z, psi)
        plt.gca().set_aspect('equal')

    return eq_ds


# todo: add plotting function for various derivatives of psi

def show_qprofiles(g_file: str, eq: Equilibrium):
    # from tokamak.formats import geqdsk
    from pleque.io._geqdsk import read
    import matplotlib.pyplot as plt

    with open(g_file, 'r') as f:
        eq_gfile = read(f)

    qpsi = eq_gfile['qpsi']
    psi_n = np.linspace(0, 1, len(qpsi))

    print(eq_gfile.keys())

    psin_axis = np.linspace(0, 1, 100)
    r = np.linspace(eq.R_min, eq.R_max, 100)
    z = np.linspace(eq.Z_min, eq.Z_max, 120)
    psi = eq.psi(R=r, Z=z)

    plt.figure()
    plt.subplot(121)
    ax = plt.gca()

    cs = ax.contour(r, z, psi, 30)
    ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    if eq._first_wall is not None:
        plt.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    ax.plot(eq._mg_axis[0], eq._mg_axis[1], 'o', color='b')
    ax.plot(eq._x_point[0], eq._x_point[1], 'x', color='r')
    ax.plot(eq._x_point2[0], eq._x_point2[1], 'x', color='r')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')
    ax.set_title(r'$\psi$')
    plt.colorbar(cs, ax=ax)

    psi_mod = eq.psi(psi_n=psin_axis)
    tor_flux = eq.tor_flux(psi_n=psin_axis)
    q_as_grad = np.gradient(tor_flux, psi_mod)

    plt.subplot(322)
    ax = plt.gca()
    ax.plot(psi_n, np.abs(qpsi), 'x', label='g-file (abs)')
    ax.plot(psin_axis, q_as_grad, '-',
            label=r'$\mathrm{d} \Phi/\mathrm{d} \psi$')
    ax.plot(psin_axis, q_as_grad, '--', label='Pleque')

    ax.legend()
    ax.set_xlabel(r'$\psi_\mathrm{N}$')
    ax.set_ylabel(r'$q$')

    plt.subplot(324)
    ax = plt.gca()
    ax.plot(tor_flux, psi_mod, label='Toroidal flux')
    ax.set_ylabel(r'$\psi$')
    ax.set_xlabel(r'$\Phi$')

    plt.subplot(326)
    ax = plt.gca()
    ax.plot(psi_n, eq.pprime(psi_n=psi_n) / 1e3)
    ax.set_xlabel(r'$\psi_\mathrm{N}$')
    ax.set_ylabel(r"$p' (\times 10^3)$")
    ax2 = ax.twinx()
    ax2.plot(psi_n, eq.ffprime(psi_n=psi_n), 'C1')
    ax2.set_ylabel(r"$ff'$")


def plot_extremes(eq: Equilibrium, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    # opoints:
    for op in eq._o_points:
        ax.plot(op[0], op[1], 'o', color='C5')

    for xp in eq._x_points:
        ax.plot(xp[0], xp[1], 'x', color='C4')


# noinspection PyTypeChecker
def plot_psi_derivatives(eq: Equilibrium):
    import matplotlib.pyplot as plt

    r = np.linspace(eq.R_min, eq.R_max, 100)
    z = np.linspace(eq.Z_min, eq.Z_max, 120)

    psi = eq.psi(R=r, Z=z)

    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(8, 4))
    ax = axs[0]
    ax.contour(r, z, psi.T, 20)
    ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    if eq._first_wall is not None:
        ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    ax.plot(eq._mg_axis[0], eq._mg_axis[1], 'o', color='b', markersize=10)
    ax.plot(eq._x_point[0], eq._x_point[1], 'x', color='r', markersize=10)
    ax.plot(eq._x_point2[0], eq._x_point2[1], 'x', color='r', markersize=10)
    psi_axis = ax
    plot_extremes(eq, psi_axis)

    def psi_xysq_func(r, z):
        return eq._spl_psi(r, z, dx=1, dy=0, grid=True) ** 2 \
               + eq._spl_psi(r, z, dx=0, dy=1, grid=True) ** 2

    psi_xysq = psi_xysq_func(r, z)
    psi_xyopt = (eq._spl_psi(r, z, dx=1, dy=1, grid=True)) ** 2
    plt.title(r'$\psi$')

    ax.set_aspect('equal')

    ax = axs[1]
    cl = ax.contour(r, z, psi_xysq.T, np.linspace(0, 0.1, 50))
    # plt.colorbar(cl)
    ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    if eq._first_wall is not None:
        ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    ax.set_title(r'$\partial_x \psi^2 + \partial_y \psi^2$')
    ax.set_aspect('equal')

    ax = axs[2]
    cl = ax.contour(r, z, psi_xyopt.T, np.linspace(0, 1, 50))
    # plt.colorbar(cl)
    ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    if eq._first_wall is not None:
        ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    ax.set_title(r'$\partial_{xy} \psi^2$')
    ax.set_aspect('equal')

    psi_xy = (eq._spl_psi(r, z, dx=1, dy=1, grid=True)) ** 2
    psi_xx = (eq._spl_psi(r, z, dx=2, dy=0, grid=True))
    psi_yy = (eq._spl_psi(r, z, dx=0, dy=2, grid=True))
    D = psi_xx * psi_yy - psi_xy

    ax = axs[3]
    cl = ax.contour(r, z, D.T, np.linspace(-50, 50, 50), cmap='PiYG')
    plt.colorbar(cl)
    ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    if eq._first_wall is not None:
        ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    ax.set_title(r'$D$')
    ax.set_aspect('equal')


# noinspection PyTypeChecker
def plot_overview(eq: Equilibrium):
    import matplotlib.pyplot as plt

    r = np.linspace(eq.R_min, eq.R_max, 100)
    z = np.linspace(eq.Z_min, eq.Z_max, 120)

    psi = eq.psi(R=r, Z=z)

    plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.contour(r, z, psi, 20)
    # plt.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    plt.plot(eq.lcfs.R, eq.lcfs.Z, label='lcfs')
    if eq._first_wall is not None:
        plt.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    plt.plot(eq._mg_axis[0], eq._mg_axis[1], 'o', color='b', markersize=10)
    plt.plot(eq._x_point[0], eq._x_point[1], 'x', color='r', markersize=10)
    plt.plot(eq._x_point2[0], eq._x_point2[1], 'x', color='r', markersize=10)
    return_axis = plt.gca()

    plt.title(r'$\psi$')
    plt.gca().set_aspect('equal')

    plt.subplot(132)
    cs = plt.contour(r, z, eq.B_pol(R=r, Z=z), 20)
    plt.clabel(cs, inline=1)
    plt.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    if eq._first_wall is not None:
        plt.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')

    plt.title(r'$B_\mathrm{pol}$')
    plt.gca().set_aspect('equal')

    plt.subplot(133)
    cs = plt.contour(r, z, eq.B_tor(R=r, Z=z), 20)
    plt.clabel(cs, inline=1)
    plt.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    if eq._first_wall is not None:
        plt.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    plt.title(r'$B_\mathrm{tor}$')
    plt.gca().set_aspect('equal')

    # number of points
    n = 100
    # module automaticaly identify the type of the input:
    midplane = eq.coordinates(r=np.linspace(-0.3, 0.3, n), theta=np.zeros(n))

    fig, axs = plt.subplots(3, 1, sharex=True)

    ax = axs[0]
    # Profile of toroidal field:
    ax.plot(midplane.r, eq.B_tor(midplane))
    # Profile of poloidal field:
    ax.plot(midplane.r, eq.B_pol(midplane))

    # # ----
    # plt.figure(figsize=(8, 4))
    # plt.subplot(131)
    # plt.pcolormesh(r, z, eq.B_R(R=r, Z=z).T)
    # if eq._first_wall is not None:
    #     plt.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    # plt.title(r'$B_\mathrm{R}$')
    # plt.gca().set_aspect('equal')
    #
    # plt.subplot(132)
    # plt.pcolormesh(r, z, eq.B_Z(R=r, Z=z).T)
    # if eq._first_wall is not None:
    #     plt.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    # plt.title(r'$B_\mathrm{Z}$')
    # plt.gca().set_aspect('equal')
    #
    # plt.subplot(133)
    # plt.pcolormesh(r, z, np.sqrt(eq.B_R(R=r, Z=z) ** 2 + eq.B_Z(R=r, Z=z) ** 2).T)
    # if eq._first_wall is not None:
    #     plt.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], 'k')
    # plt.title(r'$B_\mathrm{pol}$')
    # plt.gca().set_aspect('equal')
    #
    # psi_n = np.linspace(0, 1, 100)
    #
    # plt.figure()
    # plt.subplot(211)
    # ax = plt.gca()
    # ax.plot(psi_n, eq.pressure(psi_n=psi_n), 'C1')
    # ax.set_xlabel(r'$\psi_\mathrm{N}$')
    # ax.set_ylabel(r'$p [Pa]$', color='C1')
    #
    # ax2 = ax.twinx()
    # ax2.plot(psi_n, eq.pprime(psi_n=psi_n), color='C2')
    # ax2.set_ylabel(r"$p'$", color='C2')
    #
    # plt.subplot(212)
    # ax = plt.gca()
    # ax.plot(psi_n, eq.F(psi_n=psi_n), 'C1')
    # ax.set_xlabel(r'$\psi_\mathrm{N}$')
    # ax.set_ylabel(r'$f$ ', color='C1')
    #
    # ax2 = ax.twinx()
    # ax2.plot(psi_n, eq.ffprime(psi_n=psi_n), 'C2')
    # ax2.set_ylabel(r"$ff'$ ", color='C2')

    return return_axis


def main():
    import matplotlib.pyplot as plt

    test_case = 4
    gfile = get_test_equilibria_filenames()[test_case]
    eq = load_testing_equilibrium(test_case)

    ax = plot_overview(eq)
    # plot_extremes(eq, ax)
    # plot_psi_derivatives(eq)

    show_qprofiles(gfile, eq)

    print(eq.fluxfuncs.F)
    print(eq.fluxfuncs.__dict__)

    # Show all plots generated during tests
    plt.show()


if __name__ == '__main__':
    main()
