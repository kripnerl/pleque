from scipy.signal import argrelmin
from scipy.optimize import minimize
from scipy.integrate import trapz, cumtrapz
import pleque.utils.surfaces as surf
from pleque.utils.surfaces import points_inside_curve, find_contour
import numpy as np
import xarray as xa


def is_monotonic(f, x0, x1, n_test=10):
    """
    Test whether line connection of two points is monotonic on f.

    :param f: 2D spline `f(x[0], x[1])`
    :param x0: start point (2d) of the line
    :param x1: end point (2d) of the line
    :param n_test: number of points which are tested.
    :return: logic value
    """
    rpts = np.linspace(x0[0], x1[0], n_test)
    zpts = np.linspace(x0[1], x1[1], n_test)
    psi_test = f(rpts, zpts, grid=False)
    return np.abs(np.sum(np.sign(np.diff(psi_test)))) == n_test - 1


def minimize_in_vicinity(point, func, r_lims, z_lims):
    # minimize in the vicinity:

    # Study different methods and find the most propriate and fastest!
    bounds = ((np.max((r_lims[0], point[0] - 0.1)),
               np.min((r_lims[-1], point[0] + 0.1))),
              (np.max((z_lims[0], point[1] - 0.1)),
               np.min((z_lims[-1], point[1] + 0.1))))

    res = minimize(func, point, method='Powell', options={'xtol': 1e-7})
    # res = minimize(func, point, bounds=bounds)
    res_point = np.array((res['x'][0], res['x'][1]))
    return res_point


def find_extremes(rs, zs, psi_spl):
    """
    Find the extremes on grid given by rs and zs.
    x-points: Candidates for x-point
    o-points: Candidates for magnetic axis

    :param rs: array-like(n) R - major radius coordinate
    :param zs: array-like(m) Z - vertical coordinate
    :param psi_spl:
    :return: tuple(x-points, o-points) of arrays(N, 2)
    """

    psi = psi_spl(rs, zs)
    psi_x = psi_spl(rs, zs, dx=1, dy=0)
    psi_y = psi_spl(rs, zs, dx=0, dy=1)
    psi_xysq = psi_x ** 2 + psi_y ** 2

    # this find extremes along first and second dimension
    mins0 = tuple(argrelmin(psi_xysq, axis=0))
    mins1 = tuple(argrelmin(psi_xysq, axis=1))

    # use these values to define psi_xysq_func threshold
    # psi_diff = (np.max(psi) - np.min(psi)) ** 2
    # x_diff = ((rs[-1] - rs[0]) / len(rs)) ** 2 + ((zs[-1] - zs[0]) / len(zs)) ** 2

    def psi_xysq_func(x):
        """
        Return sum of squre of gradients of psi spline in R a Z direction.
        
        return: array
        """
        return psi_spl(x[0], x[1], dx=1, dy=0, grid=False) ** 2 \
               + psi_spl(x[0], x[1], dx=0, dy=1, grid=False) ** 2

    x_points = []
    o_points = []

    for i, (ar, az) in enumerate(zip(mins0[0], mins0[1])):
        for j, (br, bz) in enumerate(zip(mins1[0], mins1[1])):
            if ar == br and az == bz:
                r_ex = rs[ar]
                z_ex = zs[az]

                # XXX Remove bad candidates for the extreme (this is potentional trouble point):
                if psi_xysq_func((r_ex, z_ex)) > 1:  # 1e3 * dpsidx:
                    continue

                psi_xx = (psi_spl(r_ex, z_ex, dx=2, dy=0, grid=False))
                psi_yy = (psi_spl(r_ex, z_ex, dx=0, dy=2, grid=False))
                psi_xy = (psi_spl(r_ex, z_ex, dx=1, dy=1, grid=False)) ** 2

                D = psi_xx * psi_yy - psi_xy

                if D > 0:
                    o_points.append((r_ex, z_ex))
                else:
                    x_points.append((r_ex, z_ex))

    o_points = np.array(o_points)
    x_points = np.array(x_points)

    return x_points, o_points


def recognize_mg_axis(o_points, psi_spl, r_lims, z_lims, first_wall=None, mg_axis_candidate=None):
    """
    Try to recognize which o_points is the magnetic axis.
    If `mg_axis_candidate` is not specified o point is identified as a point most in the center of
    calculation area and in the first wall if specified. If the candidate is specified, magnetic axis is recognize
    as the point closest to the candidate. The position of o-point finally found by minimization of sum of square of
    the first partial derivatives of psi.

    :param o_points: array-like(N, 2)
    :param psi_spl: 2D spline psi(R,Z)
    :param r_lims: tuple(Rmin, Rmax)
    :param z_lims: tuple(Zmin, Zmax)
    :param first_wall: array-like (N, 2) specification of the first wall.
    :param mg_axis_candidate: tuple (r, z)
    :return: tuple with recognize magnetic axis point and arguments of sorted o-points
    """

    if mg_axis_candidate is None:
        r_centr = (r_lims[0] + r_lims[-1]) / 2
        z_centr = (z_lims[0] + z_lims[-1]) / 2
        # vertical distance if favoured
        op_dist = 5 * (o_points[:, 0] - r_centr) ** 2 + (o_points[:, 1] - z_centr) ** 2
    else:
        op_dist = (o_points[:, 0] - mg_axis_candidate[0]) ** 2 + (o_points[:, 1] - mg_axis_candidate[1]) ** 2
    # normalise the maximal distance to one
    op_dist = op_dist / np.max(op_dist)

    # assume that psi value has its minimum in the center (is this check really needed?
    op_psiscale = 1

    # XXX this code may be usefull for axis recognition
    # op_psiscale = psi_spln(o_points[:, 0], o_points[:, 1], grid=False)
    # op_psiscale = 1 + (op_psiscale - np.min(op_psiscale)) / (np.max(op_psiscale) - np.min(op_psiscale))

    op_in_first_wall = np.zeros_like(op_dist)
    if first_wall is not None and len(first_wall) > 2:
        mask_in = points_inside_curve(o_points, first_wall)
        op_in_first_wall[mask_in] = 1
        op_in_first_wall[not mask_in] = 1e-3

    sortidx = np.argsort(op_dist * op_psiscale * (1 - op_in_first_wall))

    o_point = o_points[sortidx[0]]

    def psi_xysq_func(x):
        """
        Return sum of squre of gradients of psi spline in R a Z direction.
        
        return: array
        """
        return psi_spl(x[0], x[1], dx=1, dy=0, grid=False) ** 2 \
               + psi_spl(x[0], x[1], dx=0, dy=1, grid=False) ** 2

    o_point = minimize_in_vicinity(o_point, psi_xysq_func, r_lims, z_lims)

    return o_point, sortidx


def recognize_x_points(x_points, mg_axis, psi_axis, psi_spl, r_lims, z_lims, psi_lcfs_candidate=None,
                       x_point_candidates=None):
    if x_points is None or len(x_points) == 0:
        return (None, None), list([])

    def psi_xysq_func(x):
        """
        Return sum of squre of gradients of psi spline in R a Z direction.
        
        return: array
        """
        return psi_spl(x[0], x[1], dx=1, dy=0, grid=False) ** 2 \
               + psi_spl(x[0], x[1], dx=0, dy=1, grid=False) ** 2

    len_diff = np.ones(x_points.shape[0])
    monotonic = np.zeros(x_points.shape[0])

    psi_xps = psi_spl(x_points[:, 0], x_points[:, 1], grid=False)

    if psi_lcfs_candidate is None:
        psi_diff = np.abs(psi_xps - psi_axis)
    else:
        psi_diff = np.abs(psi_xps - psi_lcfs_candidate)

    if x_point_candidates is not None:
        if len(np.shape(x_point_candidates)) > 1:
            len_diff = (x_point_candidates[:, 0, np.newaxis] - x_points[np.newaxis, :, 0]) ** 2 + \
                       (x_point_candidates[:, 1, np.newaxis] - x_points[np.newaxis, :, 1]) ** 2
            len_diff = np.prod(len_diff, axis=0)
        else:
            len_diff = (x_point_candidates[0] - x_points[:, 0]) ** 2 + (x_point_candidates[1] - x_points[:, 1]) ** 2
        len_diff = len_diff / np.max(len_diff)

    for i, xpoint in enumerate(x_points):
        monotonic[i] = is_monotonic(psi_spl, mg_axis, xpoint, 10)
        monotonic[i] = (1 - monotonic[i] * 1) + 1e-3

    sortidx = np.argsort(psi_diff * monotonic * len_diff)
    xp1 = x_points[sortidx[0]]

    if len(x_points) < 1:
        xp2 = x_points[sortidx[1]]

        if psi_diff[sortidx[0]] > psi_diff[sortidx[1]]:
            xp1, xp2 = xp2, xp1
            sortidx[0], sortidx[1] = sortidx[1], sortidx[0]

        xp2 = minimize_in_vicinity(xp2, psi_xysq_func, r_lims, z_lims)
    else:
        xp2 = None

    xp1 = minimize_in_vicinity(xp1, psi_xysq_func, r_lims, z_lims)

    return (xp1, xp2), sortidx


def recognize_plasma_type(x_point, first_wall, mg_axis, psi_axis, psi_spl):
    """
    Recognize whether the plasma is limited or with x-point and find point which limit the plasma (limiter point).
    In case of limiter plasma it is contact point, in case of x-point the plasma is limited by x-point.

    :param x_point: (R, Z) position of point suspected to by x-point or `None` if there is any.
    :param first_wall: array(N, 2) Points which may limit the plasma.
    :param mg_axis: (R, Z) position of the magnetic axis of plasma.
    :param psi_axis: psi on axis
    :param psi_spl: 2D (R, Z) spline with psi values
    :return: tuple of (bool, array of points) if bool is True plasma is limited, x-point otherwise.
    """
    psi_wall = psi_spl(first_wall[:, 0], first_wall[:, 1], grid=False)
    psi_wall_diff = np.abs(psi_wall - psi_axis)
    idxs_wall = np.argsort(psi_wall_diff)

    # todo: tmp solution (this is not the fastest way of doing this
    #       I would like to take x-point - mg_axis vector and check whether the point is
    #       on reliable place
    # iwall_min = np.argmin(psi_wall_diff)
    # wall_min_diff = psi_wall_diff[iwall_min]

    i = 0

    while not (i == len(idxs_wall) or is_monotonic(psi_spl, first_wall[idxs_wall[i]], mg_axis, 50)):
        i += 1
    if i == len(idxs_wall):
        iwall_min = -1
        wall_min_diff = np.inf
    else:
        iwall_min = i
        wall_min_diff = psi_wall_diff[idxs_wall[i]]

    limiter_plasma = True
    limiter_point = first_wall[idxs_wall[iwall_min]]
    if x_point is not None and (len(first_wall) < 4 or points_inside_curve([x_point], first_wall)[0]):
        diff_psi_xp = np.abs(psi_spl(*x_point, grid=False) - psi_axis)
        if diff_psi_xp < wall_min_diff or iwall_min == -1:
            limiter_plasma = False
            limiter_point = x_point

    return limiter_plasma, limiter_point


def find_close_lcfs(psi_lcfs, rs, zs, psi_spl, mg_axis, psi_axis=0):
    """
    Localize the field line at the 99.9% of psi.

    :param psi_lcfs: float
    :param rs: array(N), R axis of the grid where to find the contour
    :param zs: array(M), Z axis of the grid where to find the contour
    :param first_wall: array(N_wall, 2)
    :param psi_spl: float
    :param psi_axis: float
    :return:
    """

    new_psi_lcfs = psi_lcfs - 1e-4 * (psi_lcfs - psi_axis)

    contours = find_contour(psi_spl(rs, zs, grid=True).T, new_psi_lcfs, rs, zs)

    if contours is not None:
        for contour in contours:
            if surf.curve_is_closed(contour) and surf.points_inside_curve([mg_axis], contour):
                return contour
    return None


def find_strike_points(psi_spl, rs, zs, psi_lcfs, first_wall):
    """
    Find strike points. As a strike point is assumed any intersection iso-psi-value with the first wall.

    :param psi_spl: 2D spline
    :param rs: array(N) - R component of grid used for contour finding.
    :param zs: array(Z) - Z component of grid used for contour finding.
    :param psi_lcfs: float
    :param first_wall: array(N, 2)
    :return: array(M, 2) or None
    """

    contours = find_contour(psi_spl(rs, zs, grid=True).T, psi_lcfs, rs, zs)

    sp = []

    if contours is not None:
        for contour in contours:
            intersects = surf.intersection(contour, first_wall)
            if intersects is not None:
                sp.append(intersects)

    if len(sp) > 0:
        sp_array = np.concatenate(sp)
    else:
        sp_array = None

    return sp_array


def find_surface_step(psi_spl, psi_target, flux_surf):
    """
    Use simple down-hill algorithm to make step towards `psi_target`.

    :param psi_spl: 2D spline
    :param psi_target: float
    :param flux_surf: array(2, N)
    :return:
    """
    psi = psi_spl(flux_surf[:, 0], flux_surf[:, 1], grid=False)
    psix = psi_spl(flux_surf[:, 0], flux_surf[:, 1], grid=False, dx=1, dy=0)
    psiy = psi_spl(flux_surf[:, 0], flux_surf[:, 1], grid=False, dx=0, dy=1)

    deriv_norm = np.sqrt(psix ** 2 + psiy ** 2)
    psix = psix / (deriv_norm ** 2)
    psiy = psiy / (deriv_norm ** 2)

    flux_surf[:, 0] -= 0.99 * psix * (psi - psi_target)
    flux_surf[:, 1] -= 0.99 * psiy * (psi - psi_target)

    return flux_surf


def pprime2p(pprime, psi_ax, psi_bnd):

    coef = (psi_bnd - psi_ax)

    if isinstance(pprime, xa.DataArray):
        psi_n = pprime.psi_n
    else:
        psi_n = np.linspace(0, 1, len(pprime), endpoint=True)

    p = coef * cumtrapz(pprime, psi_n, initial=0)

    p = p - p[-1]

    if isinstance(pprime, xa.DataArray):
        return xa.DataArray(p, [psi_n], ['psi_n'])
    else:
        return p

def ffprime2f(ffprime, f0, psi_ax, psi_bnd):
    coef = (psi_bnd - psi_ax)

    if isinstance(ffprime, xa.DataArray):
        psi_n = ffprime.psi_n
    else:
        psi_n = np.linspace(0, 1, len(ffprime), endpoint=True)

    f_sq = 2 * coef * cumtrapz(ffprime, psi_n, initial=0)

    f = np.sign(f0) * np.sqrt(f_sq - f_sq[-1] + f0**2)

    if isinstance(f, xa.DataArray):
        return xa.DataArray(f, [psi_n], ['psi_n'])
    else:
        return f