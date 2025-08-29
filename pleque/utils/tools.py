import numpy as np
from scipy.interpolate import BivariateSpline


def arglis(seq):
    """Returns the indices of the Longest Increasing Subsequence in the Given List/Array"""
    n = len(seq)
    p = [0] * n
    m = [0] * (n + 1)
    l = 0
    for i in range(n):
        lo = 1
        hi = l
        while lo <= hi:
            mid = (lo + hi) // 2
            if seq[m[mid]] < seq[i]:
                lo = mid + 1
            else:
                hi = mid - 1

        new_l = lo
        p[i] = m[new_l - 1]
        m[new_l] = i

        if new_l > l:
            l = new_l

    s = []
    k = m[l]

    for i in range(l - 1, -1, -1):
        s.append(k)
        k = p[k]
    return s[::-1]


def lis(seq):
    """Returns the Longest Increasing Subsequence in the Given List/Array"""
    return [seq[i] for i in arglis(seq)]


def arg_longest_increasing_subsequence(seq):
    """Returns the indices of the Longest Increasing Subsequence in the Given List/Array"""
    return arglis(seq)


def longest_increasing_subsequence(seq):
    """Returns the Longest Increasing Subsequence in the Given List/Array"""
    return lis(seq)


def arg_longest_monotonic_subsequence(seq):
    """Returns the indices of the Longest Monotonic Subsequence in the Given List/Array"""
    arglms_p = arglis(seq)
    arglms_n = arglis(seq[::-1])

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots()
    # x_ax = np.arange(len(seq))
    # ax.plot(x_ax, seq)
    # ax.plot(x_ax[arglms_p], seq[arglms_p], color="green", linestyle="--")
    # ax.plot(x_ax[arglms_n], seq[arglms_n], color="red", linestyle="--")
    # plt.show()

    return arglms_p if len(arglms_p) > len(arglms_n) else arglms_n[::-1]


def longest_monotonic_subsequence(seq):
    """Returns the Longest Monotonic Subsequence in the Given List/Array"""
    return [seq[i] for i in arg_longest_monotonic_subsequence(seq)]


def hessian(spln: BivariateSpline, R, Z, grid=False):
    """

    :param spln: BivariateSpline in R and Z coordinates
    :type spln: scipy.optimi
    :param R:
    :param Z:
    :return: (2, 2, n) matrix of psi hessian od n points
    """
    spl_rz = (spln(R, Z, dx=1, dy=1, grid=grid)).T
    spl_rr = (spln(R, Z, dx=2, dy=0, grid=grid)).T
    spl_zz = (spln(R, Z, dx=0, dy=2, grid=grid)).T

    hess = np.array([[spl_rr, spl_rz],
                     [spl_rz, spl_zz]])

    return hess


def xp_section_vecs(hess: np.ndarray):
    """
    Compute the normalized eigenvectors of a symmetric Hessian matrix used for
    calculating the x-point sections:

    Parameters:
    hess: np.ndarray
        The symmetric Hessian matrix of psi in the x-point.

    Returns:
    tuple[np.ndarray, np.ndarray]
        A tuple containing two normalized eigenvectors.
        The first is always targeting outwards (LFS x-point plane), the second is always facing upwards
        (most commonly towards inner plasma).
    """
    # Hessian matrix is always symetric
    _, evecs = np.linalg.eigh(hess)

    evec1 = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
    evec2 = evecs[:, 1] / np.linalg.norm(evecs[:, 1])

    # Control right order of vectors and directions
    if np.abs(evec1[0]) < np.abs(evec1[1]):
        evec1, evec2 = evec2, evec1

    if evec1[0] < 0:
        evec1 = -evec1

    if evec2[1] < 0:
        evec2 = -evec2

    return evec1, evec2


def xp_sections(spln: BivariateSpline, R_xp: float, Z_xp: float, length: float = 1.0, n_points: int = 100) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
   Generates poloidal cross-sections of the planes of the x-point section (Σ_s).
   
   Args:
       spln (BivariateSpline): Spline representation of poloidal flux.
       R_xp (float): R-coordinate of the X-point.
       Z_xp (float): Z-coordinate of the X-point.
       length (float, optional): The extension length for each section along the
           eigenvectors. Defaults to 1.0.
       n_points (int, optional): Number of points for linspace, defining the
           resolution of each section. Defaults to 100.
   
   Returns:
       tuple: Tuple of four arrays containing the R and Z coordinates for each plane.
       The order of x-point planes directions (with respect to x-point) (lfs, in-plasma, hfs, out) is preserved.
   """
    hass = hessian(spln, R_xp, Z_xp)
    evecs = xp_section_vecs(hass)

    secs = tuple(
        np.array([
            np.linspace(R_xp, R_xp + evec[0] * length * sign, n_points),
            np.linspace(Z_xp, Z_xp + evec[1] * length * sign, n_points)
        ]).T
        for sign in [1, -1]
        for evec in evecs
    )

    # upper x-point configuraton
    if Z_xp > 0:
        secs = (secs[0], secs[3], secs[2], secs[1])

    return secs # noqa


def xp_vecs(spln: BivariateSpline, R, Z):
    """
    Calculate matrix of field line tracing equation in x-point and it's analytical
    representation of eigenvectors
    .
    :param spln: BivariateSpline in R and Z coordinates
    :param R:
    :param Z:
    :return:
    """

    spl_rz = spln(R, Z, dx=1, dy=1, grid=False)
    spl_rr = spln(R, Z, dx=2, dy=0, grid=False)
    spl_zz = spln(R, Z, dx=0, dy=2, grid=False)

    mat = np.array([[-spl_rz, -spl_zz],
                    [spl_rr, spl_rz]])

    evecs = np.array([[- spl_rz + np.sqrt(spl_rz ** 2 - spl_rr * spl_zz), spl_rr],
                      [- spl_rz - np.sqrt(spl_rz ** 2 - spl_rr * spl_zz), spl_rr]])
    return evecs, mat