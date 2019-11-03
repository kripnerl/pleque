
import pkg_resources
import pleque.io.readers as readers
import xarray as xr
import numpy as np

import pleque

resource_package = "pleque"
equilibria_files = [pkg_resources.resource_filename(resource_package, f) for f in
                    ['resources/baseline_eqdsk',
                     'resources/scenario_1_baseline_upward_eqdsk',
                     'resources/DoubleNull_eqdsk',
                     'resources/g13127.1050',
                     'resources/14068@1130_2kA_modified_triang.gfile',
                     'resources/g15349.1120',
                     'resources/shot8078_jorek_data.nc',
                     ]]


def get_test_cases_number():
    return len(equilibria_files)


def load_testing_equilibrium(case=0, cocos=None):
    """
    Return testing equilibrium file
    :param case -
    :return:
    """
    from pleque.io.compass import read_fiesta_equilibrium

    # cases are numbered from one... for now :-)
    res_file = get_test_equilibria_filenames()[case]
    res_limiterfile = get_test_divertor()[0]

    #    equil = read_fiesta_equilibrium(res_file)
    if 'eqdsk' in res_file or 'gfile' in res_file or '/g' in res_file or '\g' in res_file:
        # load as NetCDF
        if cocos is None:
            equil = readers.read_geqdsk(res_file)
        else:
            equil = readers.read_geqdsk(res_file, cocos=cocos)
    elif '.nc' in res_file:
        # load as gfile
        with xr.open_dataset(res_file) as ds:
            basedata = ds.load()
        fw = np.array([basedata.first_wall_R, basedata.first_wall_Z]).T
        if cocos is None:
            equil = pleque.Equilibrium(basedata=basedata, first_wall=fw)
        else:
            equil = pleque.Equilibrium(basedata=basedata, first_wall=fw, cocos=cocos)
    else:
        # note recognized:
        return None

    # eq_xr = readeqdsk_xarray(res_file)
    # limiter = np.loadtxt(res_limiterfile)
    # equil = Equilibrium(eq_xr, first_wall=limiter, spline_order=3, spline_smooth=0)

    return equil


def get_test_equilibria_filenames():
    """
    Return the list with absolute path (on given instance) to gfiles dedicated for testing.
    :return:
    """
    return equilibria_files


def get_test_divertor():
    import pkg_resources
    resource_package = "pleque"

    limiterfile = 'resources/limiter_v3_1_iba.dat'
    limiter = [pkg_resources.resource_filename(resource_package, limiterfile)]
    return limiter
