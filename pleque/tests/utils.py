
from importlib import resources

import numpy as np
import xarray as xr

import pleque
import pleque.io.readers as readers


def resource_path(package, resource):
    return str(resources.files(package).joinpath(*resource.split("/")))


resource_package = "pleque"
equilibria_files = [resource_path(resource_package, f) for f in
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
    # cases are numbered from one... for now :-)
    res_file = get_test_equilibria_filenames()[case]
    get_test_divertor()[0]

    #    equil = read_fiesta_equilibrium(res_file)
    if 'eqdsk' in res_file or 'gfile' in res_file or '/g' in res_file or r'\g' in res_file:
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
    resource_package = "pleque"

    limiterfile = 'resources/limiter_v3_1_iba.dat'
    limiter = [resource_path(resource_package, limiterfile)]
    return limiter


def synthetic_test_wall():
    """First wall matching the grid of :func:`synthetic_dataset`."""
    return np.array([
        [1.0, -1.0],
        [2.0, -1.0],
        [2.0, -0.2],
        [2.0, 0],
        [2.0, 0.2],
        [2.0, 1.1],
        [1.0, 1.1],
        [1.0, 0.2],
        [1.0, 0],
        [1.0, -0.2],
        [1.0, -1.0]
    ])


def synthetic_dataset(profiles="pprime_ffprime"):
    """
    Build a simple analytic equilibrium dataset for testing.

    psi = (R - 1.5)^2 + (Z - 0.2)^2, i.e. a circular limiter plasma with the
    magnetic axis at (1.5, 0.2) and psi_axis = 0.

    :param profiles: "pprime_ffprime" (default) provides the profile derivatives,
                     "p_f" provides the integrated pressure and F profiles,
                     "none" provides no profiles (vacuum equilibrium).
    :return: xarray.Dataset suitable for direct `Equilibrium` construction
    """
    R = np.linspace(0.9, 2.1, 15)
    Z = np.linspace(-1.15, 1.2, 30)
    R_mesh, Z_mesh = np.meshgrid(R, Z)

    psi = (R_mesh - 1.5) ** 2 + (Z_mesh - 0.2) ** 2

    F0 = 1
    psi_n = np.linspace(0.0, 1.0, 10)

    data_vars = {'psi': (['Z', 'R'], psi)}

    if profiles == "pprime_ffprime":
        data_vars['pprime'] = (['psi_n'], np.linspace(-1.0, 0.0, 10))
        data_vars['FFprime'] = (['psi_n'], np.linspace(0.0, 1.0, 10))
    elif profiles == "p_f":
        import pleque.utils.equi_tools as eq_tools

        pprime = np.linspace(-1.0, 0.0, 10)
        ffprime = np.linspace(0.0, 1.0, 10)
        # psi_axis = 0; psi_lcfs given by the contact point of synthetic_test_wall()
        psi_lcfs = (2.0 - 1.5) ** 2
        data_vars['pressure'] = (['psi_n'], np.asarray(eq_tools.pprime2p(pprime, 0.0, psi_lcfs)))
        data_vars['F'] = (['psi_n'], np.asarray(eq_tools.ffprime2f(ffprime, 0.0, psi_lcfs, F0)))
    elif profiles != "none":
        raise ValueError(f"Unknown profiles option: {profiles!r}")

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            'R': R,
            'Z': Z,
            'psi_n': psi_n,
        },
        attrs={
            "F0": F0,
        }
    )
