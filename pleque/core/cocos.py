import numpy as np


"""
COCOS coefficients:
exp_Bp
sigma_Bp
sigma_cyl
sigma_pol
sign_q
sign_pprime
"""


def cocos_coefs(cocos_idx):
    """
    Define COCOS coefficients.

    For more details see Table 1 in O. Sauer, et al., Comp. Phys. Comm. 184 (2013), p. 296

    :param cocos_idx: int, COCOS index
    :return: dict with COCOS coefficients
    """

    cocos = dict.fromkeys(['exp_Bp', 'sigma_Bp', 'sigma_cyl', 'sigma_pol', 'sign_q', 'sign_pprime'])
    # cocos = dict()

    # Check valid COCOS index:
    if cocos_idx not in range(1, 19) or cocos_idx in [9, 10]:
        raise ValueError("cocos_idx is not valid cocos coefficient")

    if cocos_idx < 10:
        cocos["exp_Bp"] = 0
    else:
        cocos["exp_Bp"] = +1

    mod_idx = np.mod(cocos_idx, 10)

    if mod_idx in [1, 2, 5, 6]:
        cocos["sigma_Bp"] = +1
        cocos["sign_pprime"] = -1
    else:
        cocos["sigma_Bp"] = -1
        cocos["sign_pprime"] = +1

    if mod_idx in [1, 3, 5, 7]:
        cocos["sigma_cyl"] = +1
    else:
        cocos["sigma_cyl"] = -1

    if mod_idx in [1, 2, 7, 8]:
        cocos["sigma_pol"] = +1
        cocos["sign_q"] = +1
    else:
        cocos["sigma_pol"] = -1
        cocos["sign_q"] = -1

    return cocos

