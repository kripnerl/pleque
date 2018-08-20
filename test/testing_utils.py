def load_testing_equilibrium():
    """
    Return testing equilibrium file
    :return:
    """
    from pleque.io.readgeqdsk import readeqdsk_xarray
    from pleque.core import Equilibrium
    import numpy as np

    #gfile = 'test_files/compu/baseline_eqdsk'
    gfile = 'test_files/compu/DoubleNull_eqdsk'
    limiterfile = 'test_files/compu/limiter_v3_1_iba.dat'

    eq_xr = readeqdsk_xarray(gfile)
    limiter = np.loadtxt(limiterfile)
    equil = Equilibrium(eq_xr, first_wall=limiter, spline_order=3, spline_smooth=0)

    return equil
