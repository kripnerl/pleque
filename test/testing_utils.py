def load_testing_equilibrium():
    """
    Return testing equilibrium file
    :return:
    """
    from pleque.io.readgeqdsk import readeqdsk_xarray
    from pleque.core import Equilibrium
    import numpy as np

    gfile = '/compass/Shared/Exchange/imrisek/MATLAB/COMPASS_U/Scenarios/scenario_1_baseline_eqdsk'
    limiterfile = 'test_files/compu/limiter_v3_1_iba.dat'

    eq_xr = readeqdsk_xarray(gfile)
    limiter = np.loadtxt(limiterfile)
    equil = Equilibrium(eq_xr, first_wall=limiter, spline_smooth=0)

    return equil
