import logging
from collections.abc import Mapping
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

import pleque
from pleque.config.settings import get_settings
from pleque.utils.equi_tools import get_psi_n_on_q

logger = logging.getLogger(__name__)

#: Named contour specifications accepted by :func:`plot_selected_contours`. Each name maps to
#: an :class:`~pleque.core.Equilibrium` property returning `Coordinates` (or None if the
#: equilibrium has no such point).
NAMED_CONTOURS = ("x_point", "secondary_x_point")


def _layer_kwargs(spec, defaults=None):
    """
    Resolve a ``bool | Mapping | None`` layer specification into plotting keyword arguments.

    :param spec: `None` or `True` selects the layer defaults, `False` disables the layer,
                 a mapping is merged over the defaults.
    :param defaults: default keyword arguments of the layer.
    :return: dict of keyword arguments, or `None` if the layer is disabled.
    """
    if spec is False:
        return None

    merged = dict(defaults) if defaults else {}

    if spec is None or spec is True:
        return merged

    if isinstance(spec, Mapping):
        merged.update(spec)
        return merged

    raise TypeError(f"A layer specification has to be a bool, a mapping or None, not {type(spec).__name__}.")


def _plot_grid(eq: "pleque.Equilibrium", coords=None):
    """Return the (R, Z) grid used by the plotting routines, reusing `coords` if given."""
    if coords is not None:
        return coords

    plot_cfg = get_settings().plotting
    return eq.grid((plot_cfg.grid_nr, plot_cfg.grid_nz), 'size')


def _resolve_contour_levels(eq: "pleque.Equilibrium", contours):
    """
    Resolve a contour specification into a sorted list of unique `psi_n` levels.

    Accepted items are described in :func:`plot_selected_contours`. Named points which the
    equilibrium does not have (e.g. a secondary X-point of a single-null configuration) are
    skipped with a warning.
    """
    if contours is None or contours is False:
        return []

    if isinstance(contours, (Number, str, pleque.Coordinates)):
        contours = [contours]

    levels = []
    for item in contours:
        if isinstance(item, bool):
            # `bool` is a `Number`; `True` would silently become the psi_n = 1 contour.
            raise TypeError("A contour has to be given explicitly; `True` is not a contour specification.")
        elif isinstance(item, str):
            if item not in NAMED_CONTOURS:
                raise ValueError(f"Unknown contour name '{item}'. Known names: {', '.join(NAMED_CONTOURS)}.")
            point = getattr(eq, item)
            if point is None:
                logger.warning("The equilibrium has no %s, the requested contour is not plotted.", item)
                continue
            levels.extend(np.atleast_1d(point.psi_n).tolist())
        elif isinstance(item, pleque.Coordinates):
            levels.extend(np.atleast_1d(item.psi_n).tolist())
        elif isinstance(item, Number):
            levels.append(float(item))
        else:
            raise TypeError(
                f"A contour has to be a number, a `Coordinates` instance or one of "
                f"{NAMED_CONTOURS}, not {type(item).__name__}."
            )

    # `ax.contour` requires strictly increasing levels.
    return sorted(set(levels))


def _plot_extremes(o_points, x_points, ax: plt.Axes | None = None,
                   all_o_points=True, all_x_points=True, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(o_points[0, 0], o_points[0, 1], 'o', color='royalblue', **kwargs)
    if all_o_points and len(o_points) > 1:
        ax.plot(o_points[1:, 0], o_points[1:, 1], 'o', color='cornflowerblue', **kwargs)

    if len(x_points) > 0:
        if all_x_points:
            ax.plot(x_points[:, 0], x_points[:, 1], '+', color='crimson', **kwargs)
        else:
            n_xpoints = np.min([len(x_points), 2])
            ax.plot(x_points[:n_xpoints, 0], x_points[:n_xpoints, 1], '+', color='crimson', **kwargs)


def _plot_debug(eq: pleque.Equilibrium, ax: plt.Axes | None = None, levels=None, colorbar=False):
    if ax is None:
        ax = plt.gca()

    plot_cfg = get_settings().plotting
    rs = np.linspace(eq.R_min, eq.R_max, plot_cfg.grid_nr)
    zs = np.linspace(eq.Z_min, eq.Z_max, plot_cfg.grid_nz)

    try:
        if levels is None:
            levels = plot_cfg.debug_contour_levels
        cl = ax.contour(rs, zs, eq._spl_psi(rs, zs).T, levels)
        if colorbar:
            plt.contour(cl)
    except Exception:
        logger.warning("Something wrong with psi spline.")

    try:
        ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], "k+-", label='first wall')
    except Exception:
        logger.warning("No first wall?!")

    try:
        ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], "C0", label='LCFS')
    except Exception:
        logger.warning("LCFS in troubles?!")

    try:
        ax.contour(rs, zs, eq._spl_psi(rs, zs).T, [eq._psi_lcfs], colors="C1", linestyles="--")
    except Exception:
        logger.warning("LCFS contour problem.")

    try:
        ax.plot(eq._o_points[:, 0], eq._o_points[:, 1], "C0o", label='o-points')
    except Exception:
        logger.warning("O-points in trouble")
    try:
        ax.plot(*eq._mg_axis, "C1o", label='mg axis')
    except Exception:
        logger.warning("mg. axis in trouble")

    try:
        ax.plot(eq._x_points[:, 0], eq._x_points[:, 1], "C2x", label='x-points')
    except Exception:
        logger.warning("X-points in trouble")

    try:
        ax.plot(eq._x_point[0], eq._x_point[1], "rx", lw=2, label='x-point')
    except Exception:
        logger.warning("THE X-point in trouble")

    try:
        ax.plot(eq._limiter_point[0], eq._limiter_point[1], "g+", lw=3, label='limiter point')
    except Exception:
        logger.warning("Limiter point is in trouble.")

    try:
        ax.plot(eq._strike_points[:, 0], eq._strike_points[:, 1], "C3+", lw=2, label='strike points')
    except Exception:
        logger.warning("Strike-points in trouble.")

    ax.set_title("DEBUG PLOT")
    ax.legend()
    ax.set_aspect("equal")


def plot_extremes(eq: pleque.Equilibrium, ax: plt.Axes | None = None,
                  all_o_points=True, all_x_points=True, **kwargs):
    """
    Mark the O-points (magnetic axis) and X-points of the equilibrium.

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param all_o_points: if False, only the magnetic axis is marked.
    :param all_x_points: if False, at most two X-points are marked.
    :param kwargs: passed to `ax.plot`.
    """
    if ax is None:
        ax = plt.gca()

    _plot_extremes(eq._o_points, eq._x_points, ax=ax,
                   all_o_points=all_o_points, all_x_points=all_x_points,
                   **kwargs)


def plot_rational_surface(eq, ax=None, q_tuple=(), linestyles="--", colors="C3", *, coords=None):
    """
    Plot and label the flux surfaces with the requested rational values of the safety factor.

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param q_tuple: iterable of (m, n) pairs; the surfaces with q = m/n are plotted.
    :param linestyles: line styles passed to `ax.contour`.
    :param colors: colours passed to `ax.contour`.
    :param coords: pre-computed (R, Z) grid `Coordinates`; a default plotting grid is used if None.
    :return: the `ContourSet` of the plotted surfaces.
    """
    if ax is None:
        ax = plt.gca()

    qarr = np.array(q_tuple)
    q = qarr[:, 0] / qarr[:, 1]

    plot_cfg = get_settings().plotting
    psi_n = get_psi_n_on_q(eq, q, max_psi_n=plot_cfg.rational_q_psi_n_max)

    i_ok = np.nonzero(psi_n)[0]

    psi_n = np.atleast_1d(psi_n)[i_ok]
    qarr = qarr[i_ok, :]

    coords = _plot_grid(eq, coords)
    mask_inlcfs = eq.in_lcfs(coords)

    psi_ns = np.ma.masked_array(coords.psi_n, np.logical_not(mask_inlcfs))

    fmt = {}
    for _psi_n, m, n in zip(psi_n, qarr[:, 0], qarr[:, 1]):
        fmt[_psi_n] = f"{m}/{n}"

    # `ax.contour` requires increasing levels; `fmt` is keyed by the level, so sorting is safe.
    cs = ax.contour(coords.R, coords.Z, psi_ns, np.sort(psi_n), colors=colors, linestyles=linestyles)
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)

    return cs


def plot_separatrix(eq, ax=None, in_first_wall=True, color="C3", lw=2, ls="-", alpha=0.5, **kwargs):
    """
    Plot the separatrix (the LCFS for a limited plasma).

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param in_first_wall: if True, the separatrix is clipped to the first wall.
    :param kwargs: passed to `ax.plot`.
    """
    if ax is None:
        ax = plt.gca()

    sep = eq.separatrix
    if in_first_wall:
        in_fw = eq.in_first_wall(sep)
        rs, zs = sep.R[in_fw], sep.Z[in_fw]
    else:
        rs, zs = sep.R, sep.Z

    return ax.plot(rs, zs, color=color, lw=lw, ls=ls, alpha=alpha, **kwargs)


def plot_first_wall(eq, ax=None, color="k", lw=2, label="First wall", fill=True,
                    facecolor="lightgrey", **kwargs):
    """
    Plot the first wall contour and shade its interior.

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param fill: if False, only the contour is drawn.
    :param facecolor: colour of the shaded interior.
    :param kwargs: passed to `ax.plot`.
    """
    if ax is None:
        ax = plt.gca()

    if fill:
        ax.fill_between(eq._first_wall[:, 0], eq._first_wall[:, 1], color=facecolor)

    return ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], color=color, lw=lw,
                   label=label, **kwargs)


def plot_lcfs(eq, ax=None, color="C1", lw=2, ls="--", **kwargs):
    """
    Plot the last closed flux surface.

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param kwargs: passed to `ax.plot`.
    """
    if ax is None:
        ax = plt.gca()

    return ax.plot(eq.lcfs.R, eq.lcfs.Z, color=color, lw=lw, ls=ls, **kwargs)


def plot_psi_contours(eq, ax=None, where="in_lcfs", alpha=1, levels=None, *, coords=None, **kwargs):
    """
    Plot equally spaced contours of the poloidal flux.

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param where: `"in_lcfs"`, `"out_lcfs"` or anything else for the whole grid.
    :param levels: number of contour levels or a sequence of psi values;
                   `plotting.psi_contour_levels` is used if None.
    :param coords: pre-computed (R, Z) grid `Coordinates`; a default plotting grid is used if None.
    :param kwargs: passed to `ax.contour`.
    :return: the `ContourSet` of the plotted contours.
    """
    if ax is None:
        ax = plt.gca()

    plot_cfg = get_settings().plotting
    if levels is None:
        levels = plot_cfg.psi_contour_levels

    coords = _plot_grid(eq, coords)
    psi = eq.psi(coords)

    mask_inlcfs = eq.in_lcfs(coords)

    if where.lower() == "in_lcfs":
        psi = np.ma.masked_array(psi, np.logical_not(mask_inlcfs))
    elif where.lower() == "out_lcfs":
        psi = np.ma.masked_array(psi, mask_inlcfs)

    cl = ax.contour(coords.R, coords.Z, psi, levels, alpha=alpha, **kwargs)

    return cl


def plot_near_sol(eq: pleque.Equilibrium, ax: plt.Axes | None = None, colors="C0", dr: float | None = None,
                  lw=0.7, ls="solid", *, coords=None, **kwargs):
    """
    Plot the first few flux surfaces in the scrape-off layer.

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param dr: radial spacing [m] of the surfaces at the outer midplane;
               `plotting.sol_spacing` is used if None.
    :param coords: pre-computed (R, Z) grid `Coordinates`; a default plotting grid is used if None.
    :param kwargs: passed to `ax.contour`.
    :return: the `ContourSet` of the plotted contours.
    """
    if ax is None:
        ax = plt.gca()

    plot_cfg = get_settings().plotting
    if dr is None:
        dr = plot_cfg.sol_spacing

    contour_out = eq.coordinates(r=eq.lcfs.r_mid[0] + dr * np.arange(1, 6), theta=np.zeros(5), grid=False)
    coords = _plot_grid(eq, coords)

    return ax.contour(coords.R, coords.Z, coords.psi, np.sort(np.squeeze(contour_out.psi)), colors=colors,
                      linewidths=lw,
                      linestyles=ls, **kwargs)


def plot_selected_contours(eq: pleque.Equilibrium, ax: plt.Axes | None = None, contours=None, *, coords=None, **kwargs):
    """
    Plot the poloidal flux contours passing through explicitly selected levels or points.

    Unlike :func:`plot_psi_contours` (masked to the plasma) and :func:`plot_separatrix`
    (clipped to the first wall), the selected contours are drawn over the whole
    computational grid. This is what makes the branches of a snowflake or double-null
    configuration visible.

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param contours: a single contour specification or an iterable of them. Each item is

                     * a number -- taken directly as `psi_n` (so ``1.0`` is the separatrix
                       contour, useful for snowflake configurations),
                     * a `Coordinates` instance -- the contour(s) passing through the
                       given point(s),
                     * ``"x_point"`` or ``"secondary_x_point"`` -- the contour through the
                       respective X-point of the equilibrium. A point the equilibrium does
                       not have is skipped with a warning.

                     A mapping ``{"levels": ..., **style}`` is accepted as well; the style
                     entries are passed to `ax.contour`.
    :param coords: pre-computed (R, Z) grid `Coordinates`; a default plotting grid is used if None.
    :param kwargs: passed to `ax.contour`.
    :return: the `ContourSet` of the plotted contours, or None if nothing was selected.
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(contours, Mapping):
        spec = dict(contours)
        contours = spec.pop("levels", None)
        kwargs = {**spec, **kwargs}

    levels = _resolve_contour_levels(eq, contours)
    if not levels:
        return None

    plot_cfg = get_settings().plotting
    kwargs.setdefault("colors", plot_cfg.selected_contour_color)
    kwargs.setdefault("linewidths", plot_cfg.selected_contour_linewidth)

    coords = _plot_grid(eq, coords)

    return ax.contour(coords.R, coords.Z, coords.psi_n, levels, **kwargs)


def plot_equilibrium(eq: pleque.Equilibrium, ax: plt.Axes | None = None, *, colorbar=False,
                     dr_sol: float | None = None, contours=None,
                     first_wall=None, separatrix=None, lcfs=None,
                     psi_contours=None, near_sol=None, extremes=None):
    """
    Plot the overview of a plasma equilibrium.

    Each layer is controlled by one argument accepting `None` (draw with the default style),
    `False` (do not draw the layer) or a mapping of keyword arguments merged over the layer
    defaults, e.g. ``lcfs={"color": "k", "lw": 1}``.

    The contours selected by `contours` are drawn first, so that they lie underneath all the
    other layers.

    :param eq: `Equilibrium` instance.
    :param ax: axis to plot into; the current axis is used if None.
    :param colorbar: if True, a colorbar of the poloidal flux contours is added.
    :param dr_sol: radial spacing [m] of the near-SOL contours. Deprecated,
                   use ``near_sol={"dr": ...}`` instead.
    :param contours: selected flux contours, see :func:`plot_selected_contours`.
    :param first_wall: first wall layer, see :func:`plot_first_wall`.
    :param separatrix: separatrix layer, see :func:`plot_separatrix`.
    :param lcfs: last closed flux surface layer, see :func:`plot_lcfs`.
    :param psi_contours: poloidal flux contour layer, see :func:`plot_psi_contours`.
    :param near_sol: near scrape-off layer contours, see :func:`plot_near_sol`.
    :param extremes: O-point and X-point markers, see :func:`plot_extremes`.
    :return: the axis the equilibrium was plotted into.
    """
    if ax is None:
        ax = plt.gca()

    # The grid is shared by all the contour-based layers so that psi is evaluated only once.
    coords = _plot_grid(eq)

    has_first_wall = eq._first_wall is not None and len(eq._first_wall) > 2

    # Drawn first, so that the selected contours lie under all the other layers.
    plot_selected_contours(eq, ax, contours, coords=coords)

    if has_first_wall:
        first_wall_kwargs = _layer_kwargs(first_wall)
        if first_wall_kwargs is not None:
            plot_first_wall(eq, ax, **first_wall_kwargs)

        separatrix_kwargs = _layer_kwargs(separatrix)
        if separatrix_kwargs is not None:
            plot_separatrix(eq, ax, **separatrix_kwargs)

    lcfs_kwargs = _layer_kwargs(lcfs)
    if lcfs_kwargs is not None:
        plot_lcfs(eq, ax, **lcfs_kwargs)

    cl = None
    psi_contours_kwargs = _layer_kwargs(psi_contours)
    if psi_contours_kwargs is not None:
        cl = plot_psi_contours(eq, ax, coords=coords, **psi_contours_kwargs)

    if colorbar:
        if cl is None:
            logger.warning("A colorbar was requested, but the psi contours are not plotted.")
        else:
            plt.colorbar(cl, ax=ax)

    near_sol_kwargs = _layer_kwargs(near_sol)
    if near_sol_kwargs is not None:
        near_sol_kwargs.setdefault("dr", dr_sol)
        plot_near_sol(eq, ax, coords=coords, **near_sol_kwargs)

    extremes_kwargs = _layer_kwargs(extremes, {"all_o_points": False, "all_x_points": False})
    if extremes_kwargs is not None:
        plot_extremes(eq, ax, **extremes_kwargs)

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    if has_first_wall:
        normalize_axis_xylim_by_first_wall(eq, ax)
    ax.set_aspect('equal')

    return ax


def normalize_axis_xylim_by_first_wall(eq, ax):
    rlim = [np.min(eq.first_wall.R), np.max(eq.first_wall.R)]
    zlim = [np.min(eq.first_wall.Z), np.max(eq.first_wall.Z)]

    margin = get_settings().plotting.axis_margin

    size = rlim[1] - rlim[0]
    rlim[0] -= size * margin
    rlim[1] += size * margin

    size = zlim[1] - zlim[0]
    zlim[0] -= size * margin
    zlim[1] += size * margin
    ax.set_xlim(*rlim)
    ax.set_ylim(*zlim)


def plot_cocos_geometry(eq: pleque.Equilibrium):
    # TODO STUB

    _fig, axs = plt.subplots(1, 2, projection='polar')

    # Top view:
    ax = axs[0]

    # Plot borders:
    phi = np.linspace(0, 2 * np.pi)
    r1 = 0.25 * np.ones_like(phi)
    r2 = 0.75 * np.ones_like(phi)

    np.linspace(0, np.pi / 4)
    # TODO

    ax.plot(phi, r1, 'k-')
    ax.plot(phi, r2, 'k-')

    # Polar cut:
    theta = np.linspace((0, 2 * np.pi))
    r = np.ones_like(theta)

    ax.plot(theta, r, 'k-')
