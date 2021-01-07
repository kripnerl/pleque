import numpy as np
from skimage import measure
import shapely.geometry as geo


def find_contour(array, level, r=None, z=None, fully_connected="low", positive_orientation="low"):
    """
    Finds contour using skimage,measure.find_contours function.

    :param array: 2d map of function values viz. skimage.measure.find_contours
    :param level: function value on the contour viz. skimage.measure.find_contours
    :param r: if none, coordinates in r.shape[0] dimension given. If r coordinates are specified, the coordinates
    are recalculated into r dimension
    :param z: if none, coordinates in z.shape[0] dimension given. If r coordinates are specified, the coordinates
    are recalculated into z dimension
    :param fully_connected: viz skimage.measure.find_contours function parameters
    :param positive_orientation: viz skimage.measure.find_contours function parameters
    :return: list of arrays with contour coordinates
    """
    # calling skimage function to get counturs
    coords = measure.find_contours(array.T, level, fully_connected=fully_connected,
                                   positive_orientation=positive_orientation)

    # if r, z coordinates are passed contour points are recalculated
    if isinstance(r, np.ndarray) or r is not None:
        for i in range(len(coords)):
            coords[i][:, 0] = coords[i][:, 0] / r.shape[0] * (r.max() - r.min()) + r.min()

    if isinstance(z, np.ndarray) or z is not None:
        for i in range(len(coords)):
            coords[i][:, 1] = coords[i][:, 1] / z.shape[0] * (z.max() - z.min()) + z.min()

    return coords


def intersection(line1, line2):
    """
    Find the intersection points of two lines.

    :param line1: array(N, 2)
    :param line2: array(N, 2)
    :return: array(N_intersect, 2) or empty array
    """
    # TODO: move this method to some utilities (!)

    l1 = geo.LineString(line1)
    l2 = geo.LineString(line2)

    intersec = l1.intersection(l2)

    # later version of shapely has 'array_interface' for empty intersection.
    # So instead of "None empty array is returned (array.size == 0)
    if hasattr(intersec, "array_interface"):
        intersec = np.atleast_2d(intersec)
    else:
        intersec = np.array([])

    return intersec


def fluxsurf_error(psi_spl, points, psi_target):
    """
    Calculate :math:`1/N \sum_i (\psi_i - \psi_\mathrm{target})^2`

    :param psi_spl: 2D spline
    :param points: array(N, 2)
    :param psi_target: float
    :return:
    """

    psi = psi_spl(points[:, 0], points[:, 1], grid=False)

    abs_err = 1 / len(psi) * np.sqrt(np.sum((psi - psi_target) ** 2))
    rel_err = abs_err / psi_target

    return np.abs(rel_err)


def add_xpoint(xp, lcfs, center):
    """
    Add x-point to boundary and roll it to be x-point first.

    :param xp: [x, y]
    :param lcfs: array(n, 2)
    :param center: [x, y]
    :return: array(n+1, 2)
    """

    xp_theta = np.arctan2(xp[1] - center[1], xp[0] - center[0])
    thetas = np.arctan2(lcfs[:, 1] - center[1], lcfs[:, 0] - center[0])
    if thetas[2] - thetas[1] < 0:
        thetas = -thetas
        xp_theta = -xp_theta

    idx = np.argmin(np.mod(thetas - xp_theta, np.pi * 2))

    new_lcfs = np.roll(lcfs[:-1, :], -idx, axis=0)
    new_lcfs = np.insert(new_lcfs, 0, xp, axis=0)

    return new_lcfs


def points_inside_curve(points, contour):
    """
    Uses skimage.measure.points_in_poly function to find whether points are inside a contour (polygon)
    :param points: 2d array (N, 2) of points coordinates viz. skimage.measure.points_in_poly
    :param contour: 2d array of contour (polygon) coordinates viz. skimage.measure.points_in_poly
    :return: array of bool
    """
    return measure.points_in_poly(points, contour)


def get_surface(equilibrium, psi, r=100, z=100, norm=True, closed=True, insidelcfs=True):
    """
    Finds points of surface with given value of psi.
    :param equilibrium: Equilibrium object
    :param psi: Value of psi to get the surface for
    :param r: If number, specifies number of points in the r dimension of the mesh. If numpy array,
    gives r grid points.
    :param z: If number, specifies number of points in the z dimension of the mesh. If numpy array,
    gives z grid points.
    :param norm: Specifies whether we are working with normalised values of psi
    :param closed: Are we looking for a closed surface?
    :param insidelcfs: Are we looking for a closed surface inside lcfs?
    :return: List of contours with surface coordinates
    """

    # if r is integer make r grid
    if not isinstance(r, np.ndarray):
        r = np.linspace(equilibrium.R_min, equilibrium.R_max, r)

    # if z is integer make z grid
    if not isinstance(z, np.ndarray):
        z = np.linspace(equilibrium.Z_min, equilibrium.Z_max, z)

    # should we work with psi or psi_n
    if norm:
        psipol = equilibrium.psi_n(R=r, Z=z)
    else:
        psipol = equilibrium.psi(R=r, Z=z)

    # find contours
    contour = find_contour(psipol, psi, r, z)

    # now we want the surfaces which enclose the magnetic acis, not some surfaces outside the vessel
    fluxsurface = []
    magaxis = np.expand_dims(equilibrium._mg_axis, axis=0)
    for i in range(len(contour)):
        # are we looking for a closed magnetic surface and is it closed?
        if closed and curve_is_closed(
                contour[i]):
            isinside = measure.points_in_poly(magaxis, contour[i])
            # surface inside lcfs has to be enclosing magnetic axis
            if insidelcfs and np.asscalar(isinside):
                fluxsurface.append(contour[i])
    return fluxsurface


def curve_is_closed(points):
    """
    Check whether the curve given by points is closed. The curve as assumed to be closed if the last point is
    close to the first one.

    :param points: array (N, 2)
    :return: boolean
    """

    return np.isclose(points[0, 0], points[-1, 0]) and np.isclose(points[0, 1], points[-1, 1])


def point_inside_fluxsurface(equilibrium, points, psi, r=100, z=100, norm=True,
                             insidelcfs=True):
    """
    Checks if a point is inside a flux surface with specified value of psi.

    :param equilibrium: Equilibrium object
    :param points: 2d numpy array (N, 2) of points coordinates
    :param psi: value of the psi on the surface
    :param r: If number, specifies number of points in the r dimension of the mesh. If numpy array,
    gives r grid points.
    :param z: If number, specifies number of points in the z dimension of the mesh. If numpy array,
    gives z grid points.
    :param norm:  Specifies whether we are working with normalised values of psi
    :param insidelcfs:  Are we looking for a closed surface inside lcfs?
    :return: array of bool
    """
    closed = True  # looking for a points inside a not closed surface is ambiguous

    contour = get_surface(equilibrium=equilibrium, psi=psi, r=r, z=z, closed=closed, norm=norm,
                          insidelcfs=insidelcfs)
    isinside = []
    for i in range(len(contour)):
        isinside.append(points_inside_curve(points, contour[i]))

    return isinside, contour


def point_in_first_wall(equilibrium, points):
    """
    Checks if points are inside first wall contour.
    :param equilibrium: Equilibrium object
    :param points: 2d numpy array (N, 2) of points coordinates
    :return:
    """

    isinside = points_inside_curve(points, equilibrium._first_wall)

    return isinside


def track_plasma_boundary(equilibrium, xp, xp_shift=1e-6, vect_no=0, phi_0=0):
    """
    Tracing one of two separatrix branches (switched by `vect_no`) which are goes around magnetic axis
    for `direction = 1` or in the opposite direction for `direction = -1`.

    :param equilibrium:
    :type equilibrium: pleque.Equilibrium
    :param xp: x-point position
    :param vect_no: (0, 1) Choose one of the eigen vectors of matrix of field line differential equation.
    :return:
    """
    from pleque.utils.tools import xp_vecs
    import numpy.linalg as la

    evecs, _ = xp_vecs(equilibrium._spl_psi, *xp)
    mg_axis = equilibrium._mg_axis

    evec = evecs[vect_no]

    # if there is obtuse angle between line connecting x-point and mg axis and
    # the separatrix branch multiply t
    vec_dir = np.sign(evec.dot(mg_axis - xp))
    if evec.dot(mg_axis - xp) < 0:
        evec *= -1

    evec /= la.norm(evec) / xp_shift

    br = equilibrium.B_R(*(xp + evec))[0]
    bz = equilibrium.B_Z(*(xp + evec))[0]

    # bpol = np.square(np.array([br, bz]))
    bpol = np.asarray([br, bz])

    direction = np.sign(evec.dot(bpol))

    if xp_shift > 0:
        stopper = 'poloidal'
    elif xp_shift < 0:
        stopper = 'z-stopper'
    else:
        raise ValueError('xp_shift should be != 0')

    coord = equilibrium.coordinates(R=xp[0] + evec[0], Z=xp[1] + evec[1], phi=phi_0)

    trace = equilibrium.trace_field_line(coord, stopper_method=stopper, in_first_wall=True, direction=direction)
    t = trace[0]
    rs = t.R
    zs = t.Z

    return t
    # rs = np.hstack((rs[-1], rs))
    # zs = np.hstack((zs[-1], zs))
    # fs = equilibrium._as_fluxsurface(rs, zs)
    # return fs
