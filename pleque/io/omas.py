import numpy as np
import omas
import xarray as xr

from pleque.core import Equilibrium

def read(ods: omas.ODS, time=None, time_unit="s"):
    """

    :param ods: OMAS object with description of `wall` and `equilibrium`
    :param time: (int, float or None) Time of required equilibria. The nearest time slice is taken. If `None`
                 the first time slice is taken.
    :param time_unit:  (str) "s" or "ms"
    :return: Instance of `Equilibrioum` from the `OMAS` object keeping the IMAS data structure.
    """

    if time_unit.lower() == "s":
        time_factor = 1.
    elif time_unit.lower() == "ms":
        time_factor = 1000.
    else:
        raise ValueError("Unknown `time_unit`.")

    try:
        shot = ods["info"]["shot"]
    except:
        shot = ods["dataset_description"]["data_entry"]["pulse"]

    if "wall" not in ods:
        try:
            # todo: (Need the latest OMAS)
            from omas.omas_physics import add_wall
            omas.add_wall(omas)
        except ImportError:
            print("The newest OMAS is required to add wall to IDS.")
        except KeyError:
            pass

    # Reading wall:
    try:
        r_wall = ods["wall"]["description_2d"][0]["limiter"]["unit"][0]["outline"]["r"]
        z_wall = ods["wall"]["description_2d"][0]["limiter"]["unit"][0]["outline"]["z"]

        limiter = np.stack((r_wall, z_wall)).T
    except ValueError:
        limiter = None

    # Times
    ods_times = ods["equilibrium"]["time"]
    if time is None:
        time_idx = 0
    else:
        time_idx = np.argmin(np.abs(np.array(ods_times) - time / time_factor))

    # Plasma boundary (LCFS)
    rbnd = ods["equilibrium"]["time_slice"][time_idx]["boundary"]["outline"]["r"]
    zbnd = ods["equilibrium"]["time_slice"][time_idx]["boundary"]["outline"]["z"]
    psibnd = ods["equilibrium"]["time_slice"][time_idx]["global_quantities"]["psi_boundary"]

    # Magnetic axis
    rmgax = ods["equilibrium"]["time_slice"][time_idx]["global_quantities"]["magnetic_axis"]["r"]
    zmgax = ods["equilibrium"]["time_slice"][time_idx]["global_quantities"]["magnetic_axis"]["z"]
    psimgax = ods["equilibrium"]["time_slice"][time_idx]["global_quantities"]["psi_axis"]

    # 1D profiles
    psi_1d = ods["equilibrium"]["time_slice"][time_idx]["profiles_1d"]["psi"]
    F = ods['equilibrium.time_slice'][time_idx]['profiles_1d.f']
    pressure = ods['equilibrium.time_slice'][time_idx]['profiles_1d.pressure']
    FFp = ods['equilibrium.time_slice'][time_idx]['profiles_1d.f_df_dpsi']
    pprime = ods['equilibrium.time_slice'][time_idx]['profiles_1d.dpressure_dpsi']
    q = ods['equilibrium.time_slice'][time_idx]['profiles_1d.q']

    psi_n = (psi_1d - psi_1d[0]) / (psibnd - psimgax)

    # 2D profiles:
    # todo: resolve this (!)
    grid_idx = 0
    gridtype = ods["equilibrium"]["time_slice"][time_idx]["profiles_2d"][0]["grid_type"]["index"]

    if gridtype != 1:
        raise ValueError("Only rectangular is supported at the moment!")
    # if 1 not in gridtype:
    #     raise ValueError("Only rectangular is supported at the moment!")
    # else:
    #     grid_idx = gridtype.index(1)


    r_grid = ods["equilibrium"]["time_slice"][time_idx]["profiles_2d"][grid_idx]["grid"]["dim1"]
    z_grid = ods["equilibrium"]["time_slice"][time_idx]["profiles_2d"][grid_idx]["grid"]["dim2"]
    psi = ods["equilibrium"]["time_slice"][time_idx]["profiles_2d"][grid_idx]["psi"]

    if psi.shape != (len(r_grid), len(z_grid)):
        psi = psi.T

    # x-array Dataset with time data:
    ds = xr.Dataset({
            'psi': (['R', 'Z'], psi),
            'pressure': (['psi_n'], pressure),
            'pprime': (['psi_n'], pprime),
            'F': (['psi_n'], F),
            'FFprime': (['psi_n'], FFp),
            'q': (['psi_n'], q),
        },
        coords={
            'time': ods_times[time_idx],
            'R': r_grid,
            'Z': z_grid,
            'psi_n': psi_n,
         },
        attrs={
            'shot' : shot,
            'time_unit' : time_unit,
        })

    eq = Equilibrium(ds, limiter)
    return eq


def write(equilibrium: Equilibrium, grid_1d=None, grid_2d=None, gridtype=1, ods=None,
          cocosio=3):
    """
    Function saving contents of equilibrium into the omas data structure.

    :param equilibrium: Equilibrium object
    :param grid_1d: Coordinate object with 1D grid (linearly spaced array over psi_n). It is used to save 1D equilibrium
                    characteristics into the omas data object. If is None, linearly spaced vector over psi_n inrange [0,1] with 200
                    points is used.
    :param grid_2d: Coordinate object with 2D grid (can be the whole reconstruction space). It is used to save 2D
                    equilibrium profiles. If None, default Equilibrium.grid() is used to generate the parameter.
    :param gridtype: Grid type specification for omas data structure. 1...rectangular.
                     (Any other grid  type is not supported at them moment!)
    :param ods: ods object to save the equilibrium into. If None, new ods object is created.
    :return:
    """
    # todo: Well for now it is just what ASCOT eats, we will extend this when we pile up more usage....
    if ods is None:
        ods = omas.ODS(cocosio=cocosio)

    if grid_1d is None:
        grid_1d = equilibrium.coordinates(psi_n=np.linspace(0, 1, 200))

    if grid_2d is None:
        grid_2d = equilibrium.grid(resolution=(1e-3, 1e-3), dim="step")

    shot_time = equilibrium.time
    if equilibrium.time_unit == "ms":
        shot_time *= 1e3
    elif equilibrium.time_unit != "s":
        print("WARNING: unknown time unit ({}) is used. Seconds will be used insted for saving.".format(
            equilibrium.time_unit))

    # shot info todo
    ods["info"]["shot"] = equilibrium.shot
    ods["dataset_description"]["data_entry"]["pulse"] = equilibrium.shot

    # fill the wall part
    ods["wall"]["ids_properties"]["homogeneous_time"] = 1
    # to MT: is this change correct?
    ods["wall"]["time"] = np.array(shot_time, ndmin=1)
    ods["wall"]["description_2d"][0]["limiter"]["unit"][0]["outline"]["r"] = equilibrium.first_wall.R
    ods["wall"]["description_2d"][0]["limiter"]["unit"][0]["outline"]["z"] = equilibrium.first_wall.Z

    ############################
    # The Equilibrium Part
    ############################
    # time slices
    ods["equilibrium"]["ids_properties"]["homogeneous_time"] = 1
    # to MT: is this change correct?
    ods["equilibrium"]["time"] = np.array(shot_time, ndmin=1)

    # Vacuum
    # todo: add vacuum Btor, not in equilibrium
    # As R0 use magnetic axis. Isn't better other definition of R0?
    # R0 = np.array((np.max(equilibrium.first_wall.R) - np.min(equilibrium.first_wall.R))/2, ndim=1)
    R0 = equilibrium.magnetic_axis.R[0]
    F0 = equilibrium.BvacR
    B0 = F0/R0*np.ones_like(ods["equilibrium"]["time"])
    ods["equilibrium"]["vacuum_toroidal_field"]["b0"] = B0  # vacuum B tor at Rmaj
    ods["equilibrium"]["vacuum_toroidal_field"]["r0"] = R0  # vacuum B tor at Rmaj

    # time slice time
    ods["equilibrium"]["time_slice"][0]["time"] = shot_time

    # plasma boundary (lcfs)
    ods["equilibrium"]["time_slice"][0]["boundary"]["outline"]["r"] = equilibrium.lcfs.R
    ods["equilibrium"]["time_slice"][0]["boundary"]["outline"]["z"] = equilibrium.lcfs.Z
    ods["equilibrium"]["time_slice"][0]["global_quantities"]["psi_boundary"] = equilibrium._psi_lcfs

    # Magnetic axis
    ods["equilibrium"]["time_slice"][0]["global_quantities"]["magnetic_axis"]["r"] = equilibrium.magnetic_axis.R[0]
    ods["equilibrium"]["time_slice"][0]["global_quantities"]["magnetic_axis"]["z"] = equilibrium.magnetic_axis.Z[0]
    ods["equilibrium"]["time_slice"][0]["global_quantities"]["psi_axis"] = equilibrium._psi_axis
    # define the 1 and 2d grids

    # 1d profiles
    ods["equilibrium"]["time_slice"][0]["profiles_1d"]["psi"] = equilibrium.psi(grid_1d)
    ods["equilibrium"]["time_slice"][0]["profiles_1d"]["rho_tor"] = equilibrium.tor_flux(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.f'] = equilibrium.F(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.pressure'] = equilibrium.pressure(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.f_df_dpsi'] = equilibrium.FFprime(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.dpressure_dpsi'] = equilibrium.pprime(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.q'] = equilibrium.q(grid_1d)

    # get surface volumes and areas
    surface_volume = np.zeros_like(grid_1d.psi)
    surface_area = np.zeros_like(grid_1d.psi)

    for i in range(grid_1d.psi.shape[0]):
        psin_tmp = grid_1d.psi_n[i]

        surface = 0

        if not psin_tmp == 1:
            coord_tmp = equilibrium.coordinates(psi_n=psin_tmp)
            surface = equilibrium._flux_surface(coord_tmp)

        elif psin_tmp == 1:
            surface = [equilibrium._as_fluxsurface(equilibrium.lcfs)]

        # todo: really 0 if open?
        # todo: fix this 'surface[0].closed' ... maybe calculate it from boundary
        if len(surface) > 0 and surface[0].closed:
            surface_volume[i] = surface[0].volume
            surface_area[i] = surface[0].area
        else:
            surface_volume[i] = 0
            surface_area[i] = 0

    ods["equilibrium"]["time_slice"][0]["profiles_1d"]["volume"] = surface_volume
    ods["equilibrium"]["time_slice"][0]["profiles_1d"]["area"] = surface_area

    # 2D profiles
    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["grid_type"]["index"] = gridtype

    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["grid"]["dim1"] = grid_2d.R
    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["grid"]["dim2"] = grid_2d.Z

    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["psi"] = equilibrium.psi(grid_2d).T
    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["b_field_tor"] = equilibrium.B_tor(grid_2d).T

    ods['equilibrium.time_slice'][0]['global_quantities.ip'] = equilibrium.I_plasma

    return ods
