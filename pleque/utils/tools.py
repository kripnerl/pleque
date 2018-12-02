def arglis(seq):
    """Returns arguments of the Longest Increasing Subsequence in the Given List/Array"""
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

from scipy.interpolate import BivariateSpline

def hessian(spln : BivariateSpline, R, Z, grid=False):
    """

    :param spln: BivariateSpline in R and Z coordinates
    :param R:
    :param Z:
    :return: (2, 2, n) matrix of psi hessian od n points
    """
    import numpy as np

    spl_rz = (spln(R, Z, dx=1, dy=1, grid=grid)).T
    spl_rr = (spln(R, Z, dx=2, dy=0, grid=grid)).T
    spl_zz = (spln(R, Z, dx=0, dy=2, grid=grid)).T

    hess = np.array([[spl_rr, spl_rz],
                     [spl_rz, spl_zz]])

    return hess

def xp_vecs(spln : BivariateSpline, R, Z):
    """
    Calculate matrix of field line tracing equation in x-point and it's analytical
    representation of eigenvectors
    .
    :param spln: BivariateSpline in R and Z coordinates
    :param R:
    :param Z:
    :return:
    """

    import numpy as np

    spl_rz = (spln(R, Z, dx=1, dy=1, grid=False))
    spl_rr = (spln(R, Z, dx=2, dy=0, grid=False))
    spl_zz = (spln(R, Z, dx=0, dy=2, grid=False))

    mat = np.array([[-spl_rz, -spl_zz],
                    [ spl_rr,  spl_rz]])

    evecs = np.array([[- spl_rz + np.sqrt(spl_rz**2 - spl_rr*spl_zz), spl_rr],
                      [- spl_rz - np.sqrt(spl_rz**2 - spl_rr*spl_zz), spl_rr]])
    return evecs, mat