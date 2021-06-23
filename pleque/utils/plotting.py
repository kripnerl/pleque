import matplotlib.pyplot as plt
import numpy as np

import pleque
from pleque.utils.equi_tools import get_psi_n_on_q


def _plot_extremes(o_points, x_points, ax: plt.Axes = None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(o_points[:, 0], o_points[:, 1], 'o', color='royalblue', **kwargs)
    ax.plot(x_points[:, 0], x_points[:, 1], '+', color='crimson', **kwargs)


def _plot_debug(eq: pleque.Equilibrium, ax: plt.Axes = None, levels=None, colorbar=False):
    if ax is None:
        ax = plt.gca()

    rs = np.linspace(eq.R_min, eq.R_max, 400)
    zs = np.linspace(eq.Z_min, eq.Z_max, 600)
    
    try:
        if levels is None:
            levels = 60
        cl = ax.contour(rs, zs, eq._spl_psi(rs, zs).T, levels)
        if colorbar:
            plt.contour(cl)
    except Exception:
        print("WARNING: Something wrong with psi spline.")

    try:
        ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], "k+-", label='first wall')
    except Exception:
        print("WARNING: No first wall?!")

    try:
        ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], "C0", label='LCFS')
    except Exception:
        print("WARNING: LCFS in troubles?!")

    try:
        ax.contour(rs, zs, eq._spl_psi(rs, zs).T, [eq._psi_lcfs], colors="C1", linestyles="--")
    except Exception:
        print("WARNING: LCFS contour problem.")

    try:
        ax.plot(eq._o_points[:, 0], eq._o_points[:, 1], "C0o", label='o-points')
    except Exception:
        print("WARNING: O-points in trouble")
    try:
        ax.plot(*eq._mg_axis, "C1o", label='mg axis')
    except Exception:
        print("WARNING: mg. axis in trouble")

    try:
        ax.plot(eq._x_points[:, 0], eq._x_points[:, 1], "C2x", label='x-points')
    except Exception:
        print("WARNING: X-points in trouble")

    try:
        ax.plot(eq._x_point[0], eq._x_point[1], "rx", lw=2, label='x-point')
    except Exception:
        print("WARNING: THE X-point in trouble")

    try:
        ax.plot(eq._limiter_point[0], eq._limiter_point[1], "g+", lw=3, label='limiter point')
    except:
        print("WARNING: Limiter point is in trouble.")

    try:
        ax.plot(eq._strike_points[:, 0], eq._strike_points[:, 1], "C3+", lw=2, label='strike points')
    except:
        print("WARNING: Strike-points in trouble.")

    ax.legend()
    ax.set_aspect("equal")


def plot_extremes(eq: pleque.Equilibrium, ax: plt.Axes = None, **kwargs):
    if ax is None:
        ax = plt.gca()

    _plot_extremes(eq._o_points, eq._x_points, ax=ax, **kwargs)


def plot_rational_surface(eq, ax, q_tuple, linestyles="--", colors="C3"):
    qarr = np.array(q_tuple)
    q = qarr[:, 0] / qarr[:, 1]

    psi_n = get_psi_n_on_q(eq, q, max_psi_n=0.95)

    i_ok = np.nonzero(psi_n)[0]

    psi_n = np.atleast_1d(psi_n)[i_ok]
    qarr = qarr[i_ok, :]

    coords = eq.grid((400, 600), 'size')
    mask_inlcfs = eq.in_lcfs(coords)

    psi_ns = np.ma.masked_array(coords.psi_n, np.logical_not(mask_inlcfs))

    fmt = {}
    for _psi_n, m, n in zip(psi_n, qarr[:, 0], qarr[:, 1]):
        fmt[_psi_n] = f"{m}/{n}"

    cs = ax.contour(coords.R, coords.Z, psi_ns, psi_n, colors=colors, linestyles=linestyles)
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)

    return cs


def plot_separatrix(eq, ax, in_first_wall=True, color="C3", lw=2, ls="-", alpha=0.5):
    sep = eq.separatrix
    if in_first_wall:
        in_fw = eq.in_first_wall(sep)
        rs, zs = sep.R[in_fw], sep.Z[in_fw]
    else:
        rs, zs = sep.R, sep.Z

    ax.plot(rs, zs, color=color, lw=lw, ls=ls, alpha=alpha)


def plot_first_wall(eq, ax):
    ax.fill_between(eq._first_wall[:, 0], eq._first_wall[:, 1], color='lightgrey')
    ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], color='k', lw=2,
            label='First wall')


def plot_lcfs(eq, ax, color="C1", lw=2, ls="--"):
    ax.plot(eq.lcfs.R, eq.lcfs.Z, color=color, lw=lw, ls=ls)


def plot_psi_contours(eq, ax, where="in_lcfs", alpha=1, **kwargs):
    coords = eq.grid((400, 600), 'size')
    psi = eq.psi(coords)

    mask_inlcfs = eq.in_lcfs(coords)

    if where.lower() == "in_lcfs":
        psi = np.ma.masked_array(psi, np.logical_not(mask_inlcfs))
    elif where.lower() == "out_lcfs":
        psi = np.ma.masked_array(psi, mask_inlcfs)

    cl = ax.contour(coords.R, coords.Z, psi, 20, alpha=alpha, **kwargs)

    return cl


def plot_near_sol(eq: pleque.Equilibrium, ax: plt.Axes, colors="C0", lw=0.7, ls="solid"):
    contour_out = eq.coordinates(r=eq.lcfs.r_mid[0] + 2e-3 * np.arange(1, 6), theta=np.zeros(5), grid=False)
    coords = eq.grid((400, 600), 'size')

    ax.contour(coords.R, coords.Z, coords.psi, np.sort(np.squeeze(contour_out.psi)), colors=colors,
               linewidths=lw,
               linestyles=ls)


def plot_equilibrium(eq: pleque.Equilibrium, ax: plt.Axes = None, colorbar=False, **kwargs):
    if ax is None:
        ax = plt.gca()

    if eq._first_wall is not None and len(eq._first_wall) > 2:
        plot_first_wall(eq, ax)
        plot_separatrix(eq, ax)

    # LCFS
    plot_lcfs(eq, ax)

    cl = plot_psi_contours(eq, ax)

    if colorbar:
        plt.colorbar(cl, ax=ax)

    plot_near_sol(eq, ax)

    #    contact = eq.strike_points
    #    ax.plot(contact.R, contact.Z, "C3+")

    plot_extremes(eq, ax)

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    if len(eq._first_wall) > 2:
        normalize_axis_xylim_by_first_wall(eq, ax)
    ax.set_aspect('equal')

    return ax


def normalize_axis_xylim_by_first_wall(eq, ax):
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
