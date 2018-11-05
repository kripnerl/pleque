
def get_test_cases_number():
    return 7

def load_testing_equilibrium(case=0):
    """
    Return testing equilibrium file
    :param case -
    :return:
    """
    from pleque.io.compass import read_fiesta_equilibrium

    # cases are numbered from one... for now :-)
    res_gfile = get_test_equilibria_filenames()[case]
    res_limiterfile = get_test_divertor()[0]

    equil = read_fiesta_equilibrium(res_gfile)

    # eq_xr = readeqdsk_xarray(res_gfile)
    # limiter = np.loadtxt(res_limiterfile)
    # equil = Equilibrium(eq_xr, first_wall=limiter, spline_order=3, spline_smooth=0)

    return equil


def get_test_equilibria_filenames():
    """
    Return the list with absolute path (on given instance) to gfiles dedicated for testing.
    :return:
    """
    import pkg_resources
    resource_package = __name__

    files = ['test_files/baseline_eqdsk',
             'test_files/scenario_1_baseline_upward_eqdsk',
             'test_files/DoubleNull_eqdsk',
             'test_files/g13127.1050',
             'test_files/_Equidisk_File__15MA_T_ped_4.5keV_513x51_44WYKU_v1_0.txt',
             'test_files/14068@1130_2kA_modified_triang.gfile',
             'test_files/g15349.1120']

    equils = []

    for f in files:
        equils.append(pkg_resources.resource_filename(resource_package, f))

    return equils


def get_test_divertor():
    import pkg_resources
    resource_package = __name__

    limiterfile = 'test_files/limiter_v3_1_iba.dat'
    limiter = [pkg_resources.resource_filename(resource_package, limiterfile)]
    return limiter
