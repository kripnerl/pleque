import os


def load_testing_equilibrium(case=1):
    """
    Return testing equilibrium file
    :return:
    """
    from pleque.io.readgeqdsk import readeqdsk_xarray
    from pleque.core import Equilibrium
    import numpy as np

    path = os.path.join(os.path.dirname(__file__), 'test_files')

    if case == 1:
        gfile = os.path.join(path, 'compu', 'baseline_eqdsk')
    elif case == 2:
        gfile = os.path.join(path, 'compu', 'scenario_1_baseline_upward_eqdsk')
    else:
        gfile = os.path.join(path, 'compu', 'DoubleNull_eqdsk')

    limiterfile = os.path.join(path, 'compu', 'limiter_v3_1_iba.dat')

    eq_xr = readeqdsk_xarray(gfile)
    limiter = np.loadtxt(limiterfile)
    equil = Equilibrium(eq_xr, first_wall=limiter, spline_order=3, spline_smooth=0)

    return equil
