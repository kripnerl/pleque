
def get_test_cases_number():
    return 3

def load_testing_equilibrium(case=0):
    """
    Return testing equilibrium file
    :param case -
    :return:
    """
    from pleque.io.compass import read_fiesta_equilibrium

    # cases are numbered from one... for now :-)
    res_gfile = get_test_equilibria()[case]
    res_limiterfile = get_test_divertor()[0]

    equil = read_fiesta_equilibrium(res_gfile, res_limiterfile)

    # eq_xr = readeqdsk_xarray(res_gfile)
    # limiter = np.loadtxt(res_limiterfile)
    # equil = Equilibrium(eq_xr, first_wall=limiter, spline_order=3, spline_smooth=0)

    return equil


def get_test_equilibria():
    """
    Return the list with absolute path (on given instance) to gfiles dedicated for testing.
    :return:
    """
    import pkg_resources
    resource_package = __name__

    files = ['test_files/compu/baseline_eqdsk',
             'test_files/compu/scenario_1_baseline_upward_eqdsk',
             'test_files/compu/DoubleNull_eqdsk']

    equils = []

    for f in files:
        equils.append(pkg_resources.resource_filename(resource_package, f))

    return equils


def get_test_divertor():
    import pkg_resources
    resource_package = __name__

    limiterfile = 'test_files/compu/limiter_v3_1_iba.dat'
    limiter = [pkg_resources.resource_filename(resource_package, limiterfile)]
    return limiter
