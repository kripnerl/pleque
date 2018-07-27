import os
import sys

import numpy as np
import xarray as xa

from pleque.core import Equilibrium

modpath = os.path.expanduser("/compass/home/kripner/Projects/pyTokamak.git")
if not modpath in sys.path:  # not to stack same paths continuously if it is already there
    sys.path.insert(0, modpath)


def load_gfile(g_file):
    from tokamak.formats import geqdsk
    eq_gfile = geqdsk.read(g_file)

    psi = eq_gfile['psi']
    r = eq_gfile['r'][:, 0]
    z = eq_gfile['z'][0, :]

    eq_ds = xa.Dataset({'psi': (['Z', 'R'], psi)},
                       coords={'R': r,
                               'Z': z})

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.contour(r, z, psi)
        plt.gca().set_aspect('equal')

    return eq_ds


def plot_over_view_2d(eq: Equilibrium):
    import matplotlib.pyplot as plt

    r = eq._basedata.R.data
    z = eq._basedata.Z.data

    psi = eq.psi(R=r, Z=z)

    print('R: {}'.format(r.shape))
    print('Z: {}'.format(z.shape))
    print('psi: {}'.format(psi.shape))

    plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.pcolormesh(r, z, psi.T)
    plt.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], label='lcfs')
    plt.plot(eq._mg_axis[0], eq._mg_axis[1], 'o', color='b')
    plt.plot(eq._x_point[0], eq._x_point[1], 'x', color='r')
    plt.plot(eq._x_point2[0], eq._x_point2[1], 'x', color='r')

    plt.title(r'$\psi$')
    plt.gca().set_aspect('equal')

    plt.subplot(132)
    plt.pcolormesh(eq._basedata.R.data, eq._basedata.Z.data, eq._basedata.psi.data)
    plt.title(r'$\psi$, original data')
    plt.gca().set_aspect('equal')

    plt.subplot(133)
    rgrid, zgrid = np.meshgrid(r, z)
    plt.pcolormesh(rgrid, zgrid, psi.T)
    plt.title(r'$\psi$ shown on grid coordinates')
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

    plt.show()


def main():
    import numpy as np

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
    eq_ds = load_gfile('/compass/Shared/Exchange/imrisek/MATLAB/COMPASS_U/Scenarios/scenario_1_baseline_eqdsk')

    # plt.pcolormesh(r_grid, z_grid, psi)
    # plt.gca().set_aspect('equal')
    # plt.colorbar()
    # plt.show()

    print(eq_ds)

    eq = Equilibrium(eq_ds)

    plot_over_view_2d(eq)


#    plt.show()



if __name__ == '__main__':
    main()
