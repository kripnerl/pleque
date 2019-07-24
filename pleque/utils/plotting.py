import matplotlib.pyplot as plt
import numpy as np

import pleque

def _plot_extremes(o_points, x_points, ax: plt.Axes = None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(o_points[:, 0], o_points[:, 1], 'o', color='royalblue', **kwargs)
    ax.plot(x_points[:, 0], x_points[:, 1], '+', color='crimson', **kwargs)


def _plot_debug(eq: pleque.Equilibrium, ax: plt.Axes = None):
    if ax is None:
        ax = plt.gca()

    rs = np.linspace(eq.R_min, eq.R_max, 200)
    zs = np.linspace(eq.Z_min, eq.Z_max, 250)

    try:
        ax.contour(rs, zs, eq._spl_psi(rs, zs).T, 60)
    except Exception:
        print("WARNING: Something wrong with psi spline.")

    try:
        ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], "k+-")
    except Exception:
        print("WARNING: No first wall?!")

    try:
        ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], "C0")
    except Exception:
        print("WARNING: LCFS in troubles?!")

    try:
        ax.contour(rs, zs, eq._spl_psi(rs, zs).T, [eq._psi_lcfs], colors="C1", linestyles="--")
    except Exception:
        print("WARNING: LCFS contour problem.")

    try:
        ax.plot(eq._o_points[:, 0], eq._o_points[:, 1], "C0o")
    except Exception:
        print("WARNING: O-points in trouble")

    try:
        ax.plot(*eq._mg_axis, "C1o")
    except Exception:
        print("WARNING: mg. axis in trouble")

    try:
        ax.plot(eq._x_points[:, 0], eq._x_points[:, 1], "C2x")
    except Exception:
        print("WARNING: X-points in trouble")

    try:
        ax.plot(eq._x_point[0], eq._x_point[1], "rx", lw=2)
    except Exception:
        print("WARNING: THE X-point in trouble")

    try:
        ax.plot(eq._limiter_point[0], eq._limiter_point[1], "g+", lw=3)
    except:
        print("WARNING: Limiter point is in trouble.")

    try:
        ax.plot(eq._strike_points[:, 0], eq._strike_points[:, 1], "C3+", lw=2)
    except:
        print("WARNING: Strike-points in trouble.")

    ax.set_aspect("equal")


def plot_extremes(eq: pleque.Equilibrium, ax: plt.Axes = None, **kwargs):
    if ax is None:
        ax = plt.gca()

    _plot_extremes(eq._o_points, eq._x_points, ax=ax, **kwargs)


def plot_equilibrium(eq: pleque.Equilibrium, ax: plt.Axes = None):
    if ax is None:
        ax = plt.gca()

    if eq._first_wall is not None and len(eq._first_wall) > 2:
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

    contour_out = eq.coordinates(r=eq.lcfs.r_mid[0] + 2e-3 * np.arange(1, 11), theta=np.zeros(10), grid=False)

    ax.contour(coords.R, coords.Z, psi_in, 20)

    # todo: psi should be 1-d (!) resolve this
    ax.contour(coords.R, coords.Z, psi, np.sort(np.squeeze(contour_out.psi)), colors='C0')

    #    contact = eq.strike_points
    #    ax.plot(contact.R, contact.Z, "C3+")

    op = eq.magnetic_axis
    ax.plot(op.R, op.Z, "C0o")


    psi_lcfs = eq._psi_lcfs
    z0 = eq._mg_axis[1]

    # ax.pcolormesh(coords.R, coords.Z, mask_in)
    # ax.pcolormesh(rs, zs, mask_inlimiter)
    # ax.pcolormesh(coords.R, coords.Z, mask_insol)

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    if len(eq._first_wall) > 2:
        rlim = [np.min(eq.first_wall.R), np.max(eq.first_wall.R)]
        zlim = [np.min(eq.first_wall.Z), np.max(eq.first_wall.Z)]

        size = rlim[1] - rlim[0]
        rlim[0] -= size / 12
        rlim[1] += size / 12

        size = zlim[1] - zlim[0]
        zlim[0] -= size / 12
        zlim[1] += size / 12
        ax.set_xlim(*rlim)
        ax.set_ylim(*zlim)
    ax.set_aspect('equal')

    return ax


def plot_cocos_geometry(eq: pleque.Equilibrium):
    # TODO STUB

    fig, axs = plt.subplots(1, 2, projection='polar')

    # Top view:
    ax = axs[0]

    # Plot borders:
    phi = np.linspace(0, 2 * np.pi)
    r1 = 0.25 * np.ones_like(phi)
    r2 = 0.75 * np.ones_like(phi)

    phi_direction = np.linspace(0, np.pi / 4)
    # TODO

    ax.plot(phi, r1, 'k-')
    ax.plot(phi, r2, 'k-')

    # Polar cut:
    theta = np.linspace((0, 2 * np.pi))
    r = np.ones_like(theta)

    ax.plot(theta, r, 'k-')
