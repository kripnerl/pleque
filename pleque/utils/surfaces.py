import numpy as np
from skimage import measure


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


def point_inside_curve(points, contour):
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
        if closed and contour[i][0, 0] == contour[i][-1, 0] and contour[i][0, 1] == contour[i][-1, 1]:
            isinside = measure.points_in_poly(magaxis, contour[i])
            # surface inside lcfs has to be enclosing magnetic axis
            if insidelcfs and np.asscalar(isinside):
                fluxsurface.append(contour[i])
    return fluxsurface


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
        isinside.append(point_inside_curve(points, contour[i]))

    return isinside, contour


def point_in_first_wall(equilibrium, points):
    """
    Checks if points are inside first wall contour.
    :param equilibrium: Equilibrium object
    :param points: 2d numpy array (N, 2) of points coordinates
    :return:
    """

    isinside = point_inside_curve(points, equilibrium._first_wall)

    return isinside
