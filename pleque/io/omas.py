from pleque.core import Coordinates, Equilibrium

from test.testing_utils import load_testing_equilibrium
import omas
import numpy as np
def write(equilibrium: Equilibrium, grid_1d = None, grid_2d=None, gridtype=1, ods = None, time = np.array(0,ndmin=1), cocosio=3):
    """
    Function saving contents of equilibrium into the omas data structure.
    :param equilibrium: Equilibrium object
    :param grid_1d: Coordinate object with 1D grid (linearly spaced array over psi_n). It is used to save 1D equilibrium
    characteristics into the omas data object. If is None, linearly spaced vector over psi_n inrange [0,1] with 200
    points is used.
    :param grid_2d: Coordinate object with 2D grid (can be the whole reconstruction space). It is used to save 2D
    equilibrium profiles. If None, default Equilibrium.grid() is used to generate the parameter.
    :param gridtype: Grid type specification for omas data structure. 1...rectangular
    :param ods: ods object to save the equilibrium into. If None, new ods object is created.
    :return:
    """
    # todo: Well for now it is just what ASCOT eats, we will extend this when we pile up more usage....
    if ods is None:
        ods = omas.ODS(cocosio=cocosio)

    if grid_1d is None:
        grid_1d = equilibrium.coordinates(psi_n = np.linspace(0,1,200))

    if grid_2d is None:
        grid_2d = equilibrium.grid(resolution=(1e-3, 1e-3), dim="step")


    # shot info todo
    ods["info"]["shot"] = 0

    # fill the wall part
    ods["wall"]["ids_properties"]["homogeneous_time"] = 1
    ods["wall"]["time"] = np.array(time, ndmin=1)
    ods["wall"]["description_2d"][0]["limiter"]["unit"][0]["outline"]["r"] = equilibrium.first_wall.R
    ods["wall"]["description_2d"][0]["limiter"]["unit"][0]["outline"]["z"] = equilibrium.first_wall.Z

    #
    ############################
    # The Equilibrium Part
    ############################
    #time slices
    ods["equilibrium"]["ids_properties"]["homogeneous_time"] = 1
    ods["equilibrium"]["time"] = np.array(time, ndmin=1)


    #vacuum
    #todo: add vacuum Btor, not in equilibrium
    ods["equilibrium"]["vacuum_toroidal_field"]["b0"] = np.array([5]) # vacuum B tor at Rmaj
    ods["equilibrium"]["vacuum_toroidal_field"]["r0"] = 0.89 # vacuum B tor at Rmaj

    #time slice time
    ods["equilibrium"]["time_slice"][0]["time"] = time

    #plasma boundary (lcfs)
    ods["equilibrium"]["time_slice"][0]["boundary"]["outline"]["r"] = equilibrium.lcfs.R
    ods["equilibrium"]["time_slice"][0]["boundary"]["outline"]["z"] = equilibrium.lcfs.Z
    ods["equilibrium"]["time_slice"][0]["global_quantities"]["psi_boundary"] = equilibrium._psi_lcfs

    #Magnetic axis
    ods["equilibrium"]["time_slice"][0]["global_quantities"]["magnetic_axis"]["r"] = equilibrium.magnetic_axis.R[0]
    ods["equilibrium"]["time_slice"][0]["global_quantities"]["magnetic_axis"]["z"] = equilibrium.magnetic_axis.Z[0]
    ods["equilibrium"]["time_slice"][0]["global_quantities"]["psi_axis"] = equilibrium._psi_axis
    #define the 1 and 2d grids


    #1d profiles
    ods["equilibrium"]["time_slice"][0]["profiles_1d"]["psi"] = equilibrium.psi(grid_1d)
    ods["equilibrium"]["time_slice"][0]["profiles_1d"]["rho_tor"] = equilibrium.tor_flux(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.f'] = equilibrium.fpol(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.pressure'] = equilibrium.pressure(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.f_df_dpsi'] = equilibrium.ffprime(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.dpressure_dpsi'] = equilibrium.pprime(grid_1d)
    ods['equilibrium.time_slice'][0]['profiles_1d.q']  =equilibrium.q(grid_1d)

    #get surface volumes and areas
    surface_volume = np.zeros_like(grid_1d.psi)
    surface_area = np.zeros_like(grid_1d.psi)

    for i in range(grid_1d.psi.shape[0]):
        psin_tmp = grid_1d.psi_n[i]
        if not psin_tmp == 1:
            coord_tmp = equilibrium.coordinates(psi_n = psin_tmp)
            surface = equilibrium.flux_surface(coord_tmp)

        elif psin_tmp == 1:
            from pleque.fluxsurface import FluxSurface
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

    #2D profiles
    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["grid_type"]["index"]=gridtype

    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["grid"]["dim1"] = grid_2d.R
    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["grid"]["dim2"] = grid_2d.Z

    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["psi"] = equilibrium.psi(grid_2d).T
    ods["equilibrium"]["time_slice"][0]["profiles_2d"][0]["b_field_tor"] = equilibrium.B_tor(grid_2d).T

    #todo: plasma current is not in equilibrium yet
    try:
        ods['equilibrium.time_slice'][0]['global_quantities.ip'] = equilibrium.I_plasma
    except AttributeError:
        ods['equilibrium.time_slice'][0]['global_quantities.ip'] = 2e6

    return ods