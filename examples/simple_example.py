import numpy as np

from pleque.core import Equilibrium
from pleque.tests.utils import load_testing_equilibrium, get_test_equilibria_filenames
from pleque.io.jet import reader_jet

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

    return return_axis


def main():
    import matplotlib.pyplot as plt

    tokamak = 'JET'

    if tokamak == 'JET':
        eq = reader_jet.sal_jet(92400)
    elif tokamak == 'COMPASS-U':
        test_case = 0
        gfile = get_test_equilibria_filenames()[test_case]
        eq = load_testing_equilibrium(test_case)
    else: 
        test_case = 0
        gfile = get_test_equilibria_filenames()[test_case]
        eq = load_testing_equilibrium(test_case)
    
    eq.plot_overview()
    plot_extremes(eq)

    surfaces = []
    vols = []
    dvols = []
    psi_ns = np.linspace(0.1, 0.9, 200)
    for psi_n in psi_ns:
        sf = eq._flux_surface(psi_n=psi_n)[0]
        surfaces.append(sf)
        vols.append(sf.volume)
        dvols.append(sf.diff_volume)

    dvols2 = np.gradient(vols, psi_ns)*np.abs(eq._diff_psi_n)

    plt.figure()
    plt.plot(psi_ns, dvols, label='computed')
    plt.plot(psi_ns, dvols2, label='shapely')
    plt.legend()
    plt.show()


    print(eq.fluxfuncs.F)
    print(eq.fluxfuncs.__dict__)

    # Show all plots generated during tests
    plt.show()


if __name__ == '__main__':
    main()
