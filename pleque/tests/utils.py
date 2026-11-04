
import itertools
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


# Line-current model of a lower single null with a second (inactive) X-point 1 m above the lower one, mimicking
# COMPASS-U downward-VDE equilibria: (R, Z, strength, core radius) of a vertical chain of plasma filaments and of
# two divertor coils outside the wall. psi = sum(strength / 2 * ln(dR^2 + dZ^2 + core^2)) has its minimum on the
# axis and X-points at (0.72, -0.546) and (0.72, 0.452): like in the COMPASS-U case, the lower X-point is 4 mm off
# the nearest grid node, which lies only 2 mm from a point 1 m below the upper X-point.
_X_POINT_NEAR_DOME_FILAMENTS = (
    *((0.90, z, 0.1 / 7, 0.10) for z in np.linspace(-0.35, 0.05, 7)),
    (0.6404303, -0.73, 0.05442506, 0.02),
    (0.64948179, 0.75, 0.06265461, 0.02),
)


def _filament_psi(r, z):
    return sum(0.5 * c * np.log((r - rc) ** 2 + (z - zc) ** 2 + a ** 2) for rc, zc, c, a in _X_POINT_NEAR_DOME_FILAMENTS)


def _filament_psi_grad(x):
    r, z = x
    dr = dz = 0.0
    for rc, zc, c, a in _X_POINT_NEAR_DOME_FILAMENTS:
        d2 = (r - rc) ** 2 + (z - zc) ** 2 + a ** 2
        dr += c * (r - rc) / d2
        dz += c * (z - zc) / d2
    return np.array([dr, dz])


def _filament_critical_point(guess):
    from scipy.optimize import fsolve

    return fsolve(_filament_psi_grad, guess, xtol=1e-13)


def _wall_with_dome(dome_top_z, dome_r=(0.70, 0.74), step=1e-3):
    """Closed wall polygon with a flat-topped divertor dome, densely sampled with `step` [m]."""
    r_in, r_out, z_bot, z_top = 0.50, 1.30, -0.70, 0.60
    corners = np.array([
        [r_in, z_top], [r_out, z_top], [r_out, z_bot],
        [dome_r[1] + 0.02, z_bot], [dome_r[1], dome_top_z], [dome_r[0], dome_top_z], [dome_r[0] - 0.02, z_bot],
        [r_in, z_bot], [r_in, z_top],
    ])
    segments = []
    for p0, p1 in itertools.pairwise(corners):
        n = max(int(np.ceil(np.hypot(*(p1 - p0)) / step)), 1)
        segments.append(p0 + np.linspace(0, 1, n, endpoint=False)[:, np.newaxis] * (p1 - p0))
    return np.concatenate([*segments, corners[-1:]])


def synthetic_x_point_near_dome(dome_gap):
    """
    Synthetic lower-single-null equilibrium with the lower X-point `dome_gap` [m] above a divertor dome.

    psi is a line-current (2D) model of the plasma and two divertor coils on a 61 x 71 (R x Z) grid, with the lower
    X-point at (0.72, -0.546) and a second, inactive X-point about 1 m above it. This geometry reproduces two former
    initialisation failures seen on COMPASS-U MEQ VDE simulations: the refinement of the lower X-point candidate
    jumped to the upper X-point, and a dome a few mm below the X-point (private flux region, psi_n slightly below 1)
    was taken for the limiter. A negative `dome_gap` puts the dome top above the X-point, i.e. the plasma is limited
    by the dome.

    :param dome_gap: vertical distance [m] of the lower X-point above the dome top
    :return: tuple (basedata xarray.Dataset, first_wall array (N, 2), truth dict with the `mg_axis`,
             `x_points` (lower, upper), `psi_axis` and `psi_x_point` of the analytic model)
    """
    R = np.linspace(0.30, 1.50, 61)
    Z = np.linspace(-0.75, 0.65, 71)
    R_mesh, Z_mesh = np.meshgrid(R, Z)
    psi = _filament_psi(R_mesh, Z_mesh)

    mg_axis = _filament_critical_point((0.9, -0.05))
    x_points = np.array([_filament_critical_point((0.72, -0.55)), _filament_critical_point((0.72, 0.45))])

    psi_n = np.linspace(0, 1, 19)
    basedata = xr.Dataset(
        {
            "psi": (("Z", "R"), psi),
            "F": ("psi_n", 0.9 + 0.02 * (1 - psi_n)),
            "pressure": ("psi_n", 1e4 * (1 - psi_n) ** 2),
            "F0": 0.9,
        },
        coords={"R": R, "Z": Z, "psi_n": psi_n},
    )
    first_wall = _wall_with_dome(x_points[0, 1] - dome_gap)
    truth = {
        "mg_axis": mg_axis,
        "x_points": x_points,
        "psi_axis": float(_filament_psi(*mg_axis)),
        "psi_x_point": float(_filament_psi(*x_points[0])),
    }
    return basedata, first_wall, truth


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
