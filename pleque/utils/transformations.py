import numpy as np

def psi_n2psi(psi_n, psi_axis, psi_lcfs):
    '''
    Calculate psi for 1D
    :param psi_n: R
    :param psi_axis: psi axis,
    :param psi_lcfs: LCFS
    :return: psi
    '''
    res = psi_axis + psi_n * (psi_lcfs - psi_axis)
    return res

def psi_2psi_n(psi, psi_axis, psi_lcfs):
    '''
    Calculate psi_n for high (>1) dimensions
    :param psi:       psi
    :param psi_axis:    psi axis
    :param psi_lcfs:    LCFS
    :return:
    '''
    res = (psi - psi_axis) / (psi_lcfs - psi_axis)
    return res


def distance_1d(xt1, xt2):
    '''
    1D Distance(s) between two points (pares)
    :param xt1: R_1
    :param xt2: R_2
    :return: distance between these points (with proper sign)
    '''

    res = (xt1 - xt2)
    return res

def distance_2d(xt1, zt1, xt2, zt2):
    '''
    2D Distance(s) between two points (pares)
    :param xt1: R_1
    :param zt1: Z_1
    :param xt2: R_2
    :param zt2: Z_2
    :return: distance between these points
    '''

    res = np.sqrt((xt1 - xt2) ** 2 + (zt1 - zt2) ** 2)
    return res

def distance_3d(xt1, yt1, zt1, xt2, yt2, zt2):
    '''

    3D Distance(s) between two points (pares)
    :param xt1: R_1
    :param yt1: e.g. psi_1
    :param zt1: Z_1
    :param xt2: R_2
    :param yt1: e.g. psi_2
    :param zt2: Z_2
    :return: distance between these points
    '''

    res = np.sqrt((xt1 - xt2) ** 2 + (yt1 - yt2) ** 2 +(zt1 - zt2) ** 2)
    return res

def angle(xt1, zt1, xt2, zt2):
    '''
    Angle(s) between the x-axis and given point(s)
    :param xt1: R_1
    :param zt1: Z_1
    :param xt2: R_2
    :param zt2: Z_2
    :return: inclination angles
    '''

    res = np.arctan2(zt1 - zt2, xt1 - xt2)
    return res


def angle_coef(xt1, zt1, xt2, zt2, coef):
    '''

    Angle(s) between the x-axis and given point(s) with a "coefficient"
    :param xt1: R_1
    :param zt1: Z_1
    :param xt2: R_2
    :param zt2: Z_2
    :param coef:
    :return: inclination angle(s)
    '''

    res = angle(xt1, coef * zt1, xt2, coef * zt2)
    return res

def fmultiply(xt1, xt2):
    '''
    Muliplication
    :param xt1: first multiplier
    :param xt2: second multiplier
    :return: I assume noe explanation is needed :)
    '''

    res= xt1 * xt2
    return res

def lcos(r_m, x1_input, x2_input):
    '''
    Supporting function
    :param r_m:
    :param x1_input:
    :param x2_input:
    :return:
    '''

    res= r_m + x1_input * np.cos(x2_input)
    return res

def lsin(r_m, x1_input, x2_input):
    '''
    Supporting function
    :param r_m:
    :param x1_input:
    :param x2_input:
    :return:
    '''

    res= r_m + x1_input * np.sin(x2_input)
    return res