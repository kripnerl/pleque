def load_testing_equilibrium(case=1):
    """
    Return testing equilibrium file
    :return:
    """
    from pleque.io.readgeqdsk import readeqdsk_xarray
    from pleque.core import Equilibrium
    import numpy as np
    import pkg_resources

    resource_package = __name__

    if case == 1:
        gfile = 'test_files/compu/baseline_eqdsk'
    elif case == 2:
        gfile = 'test_files/compu/scenario_1_baseline_upward_eqdsk'
    else:
        gfile = 'test_files/compu/DoubleNull_eqdsk'

    limiterfile = 'test_files/compu/limiter_v3_1_iba.dat'

    res_gfile = pkg_resources.resource_filename(resource_package, gfile)
    res_limiterfile = pkg_resources.resource_filename(resource_package, limiterfile)

    eq_xr = readeqdsk_xarray(res_gfile)
    limiter = np.loadtxt(res_limiterfile)
    equil = Equilibrium(eq_xr, first_wall=limiter, spline_order=3, spline_smooth=0)

    return equil
