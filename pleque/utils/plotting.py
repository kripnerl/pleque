import matplotlib.pyplot as plt
import numpy as np

from pleque import Equilibrium

def _plot_extremes(o_points, x_points, ax: plt.Axes = None):

    if ax is None:
        ax = plt.gca()

    ax.plot(o_points[:, 0], o_points[:, 1], 'o', color='royalblue')
    ax.plot(x_points[:, 0], x_points[:, 1], 'x', color='coral')

def plot_extremes(eq: Equilibrium, ax: plt.Axes = None):

    if ax is None:
        ax = plt.gca()

    _plot_extremes(eq._o_points, eq._x_points, ax=ax)


def plot_equilibrium(eq: Equilibrium, ax: plt.Axes = None):
    if ax is None:
        ax = plt.gca()

    if eq._first_wall is not None:
        ax.fill_between(eq._first_wall[:, 0], eq._first_wall[:, 1], color='lightgrey')
        ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], color='k', lw=2,
                label='First wall')
    # separatrix
    ax.plot(eq.lcfs.R, eq.lcfs.Z, color='C1', lw=2, ls='--')

    coords = eq.grid((400, 600), 'size')
    rs, zs = coords.mesh()

    psi = eq.psi(coords)

    mask_inlcfs = eq.in_lcfs(coords)
    mask_inlimiter = eq.in_first_wall(coords)
    mask_insol = mask_inlcfs ^ mask_inlimiter

    psi_in = np.ma.masked_array(psi, np.logical_not(mask_inlcfs))
    psi_out = np.ma.masked_array(psi, mask_inlcfs)
    # ax.pcolormesh(coords.R, coords.Z, psi_in, shading='gouraud')

    ax.contourf(coords.R, coords.Z, psi_in, 100)
    ax.contour(coords.R, coords.Z, psi_out, 20)

    psi_lcfs = eq._psi_lcfs
    z0 = eq._mg_axis[1]

    # ax.pcolormesh(coords.R, coords.Z, mask_in)
    # ax.pcolormesh(rs, zs, mask_inlimiter)
    # ax.pcolormesh(coords.R, coords.Z, mask_insol)

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    ax.set_aspect('equal')
