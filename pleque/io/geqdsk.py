import numpy as np

import pleque
from ._geqdsk import read_as_equilibrium
from ._geqdsk import write as write_geqdsk


def read(file, cocos=3):
    """
    Read the eqdsk file and open it as `Equilibrium`.

    :param file: str, name of file with equilibrium.
    :param cocos: Tokamak coordinates convension. Default cocos = 3 (EFIT).
    :return:
    """
    with open(file, 'r') as f:
        eq = read_as_equilibrium(f, cocos)

    return eq


def basedata_to_dict(equilibrium: pleque.Equilibrium, cocos_out=13):
    data = dict.fromkeys(['nx', 'ny', 'rdim', 'zdim', 'rcentr', 'bcentr', 'rleft', 'zmid', 'rmagx', 'zmagx',
                          'simagx', 'sibdry', 'cpasma', 'F', 'pres', 'q', 'psi'])

    basedata = equilibrium._basedata

    psi_factor = 1
    cocos = equilibrium.cocos

    if cocos < 10 < cocos_out:
        psi_factor = 2 * np.pi
    elif cocos > 10 > cocos_out:
        psi_factor = 1 / (2 * np.pi)

    R = basedata.R.values
    Z = basedata.Z.values

    r0 = (np.max(R) + np.min(R)) / 2

    nx = len(R)
    ny = len(Z)

    grid_1d = equilibrium.coordinates(psi_n=np.linspace(0, 1, nx))

    data['nx'] = nx
    data['ny'] = ny

    data['rdim'] = np.max(R) - np.min(R)
    data['zdim'] = np.max(Z) - np.min(Z)
    data['rcentr'] = r0

    if "F0" in basedata:
        F0 = basedata.F0.values.item()
    else:
        try:
            F0 = basedata.F.values[-1]
        except AttributeError:
            F0 = equilibrium.F0

    data['bcentr'] = F0 / r0
    data['rleft'] = np.min(R)
    data['zmid'] = (np.max(Z) + np.min(Z)) / 2

    if "mg_axis" in basedata:
        data['rmagx'] = basedata.mg_axis.values[0]
        data['zmagx'] = basedata.mg_axis.values[1]
    else:
        data['rmagx'] = equilibrium.magnetic_axis.R[0]
        data['zmagx'] = equilibrium.magnetic_axis.Z[0]

    if "psi_axis" in basedata:
        data['simagx'] = basedata.psi_axis.values.item() * psi_factor
    else:
        data['simagx'] = equilibrium._psi_axis * psi_factor

    if "psi_lcfs" in basedata:
        data['sibdry'] = basedata.psi_axis.values.item() * psi_factor
    else:
        data['sibdry'] = equilibrium._psi_lcfs * psi_factor

    if "Ip" in basedata:
        data['cpasma'] = basedata.Ip.values.item()
    else:
        data['cpasma'] = equilibrium.I_plasma

    # Boundary:
    if "R_lcfs" in basedata and "Z_lcfs" in basedata:
        bnd_r = basedata.R_lcfs.values[:-1]
        bnd_z = basedata.Z_lcfs.values[:-1]
    else:
        bnd_r = equilibrium.lcfs.R[:-1]
        bnd_z = equilibrium.lcfs.Z[:-1]

    # todo:  maybe calculate the closest point to x-point
    ind = np.argmin(bnd_z).item()

    bnd_r = np.roll(bnd_r, -ind)
    bnd_r = np.append(bnd_r, bnd_r[0])
    bnd_z = np.roll(bnd_z, -ind)
    bnd_z = np.append(bnd_z, bnd_z[0])

    data["rbdry"] = bnd_r
    data["zbdry"] = bnd_z

    # 1d profiles:
    if "F" in basedata:
        data['F'] = basedata.values
    else:
        data['F'] = equilibrium.F(grid_1d, grid=False)

    if "FFprime" in basedata:
        data["FFprime"] = basedata.FFprime.values / psi_factor
    else:
        data['FFprime'] = equilibrium.FFprime(grid_1d, grid=False) / psi_factor

    if 'pressure' in basedata:
        data['pres'] = basedata.pressure.values
    else:
        data['pres'] = equilibrium.pressure(grid_1d, grid=False)

    if "pprime" in basedata:
        data['pprime'] = basedata.pprime.values / psi_factor
    else:
        data['pprime'] = equilibrium.pprime(grid_1d, grid=False) / psi_factor

    if "q" in basedata:
        data['q'] = basedata.q.values
    else:
        data['q'] = equilibrium.abs_q(grid_1d, grid=False)

    # todo: test tranformatiom
    data['psi'] = basedata.psi.T.values * psi_factor

    # Tokamak wall:
    if "first_wall" in basedata:
        data['rlim'] = basedata.first_wall.isel(ndim=0).values
        data['zlim'] = basedata.first_wall.isel(ndim=1).values
    else:
        data['rlim'] = equilibrium.first_wall.R
        data['zlim'] = equilibrium.first_wall.Z

    return data


def write(equilibrium: pleque.Equilibrium, file, nx=64, ny=128, nbdry=200, label=None, cocos_out=3,
          q_positive=True, use_basedata=False):
    """
    Write a GEQDSK equilibrium file.

    :param equilibrium: pleque.Equilibrum
    :param file:cocos = equilibrium.cocos str, file name
    :param nx: int, R-dimension
    :param ny: int, Z-dimension
    :param nbdry: int, None - Maximal number of points used to describe boundary (LCFS).
                 If None, default number obtained by FluxFunc will be used.
                 Note: Actual number number of points can be in (n-bry/2, n-bry)
    :param label: str, max 11 characters long text added on the beginning of g-file (default is PLEQUE)
    :param cocos_out: At the moment only perform 2pi normalization (!) (TODO)
    :param q_positive: always save q positive
    :param use_basedata: if True input quantities are used. If False splines and calculated values are used.
    :return:
    """

    """
    A dictionary containing:
      nx, ny        Number of points in R (x), Z (y)
      rdim, zdim    Sizes of the R,Z dimensions
      rcentr        Reference value of R
      bcentr        Vacuum toroidal magnetic field at rcentr
      rleft         R at left (inner) boundary
      zmid          Z at middle of domain
      rmagx, zmagx  R,Z at magnetic axis (O-point)
      simagx        Poloidal flux psi at magnetic axis
      sibdry        Poloidal flux psi at plasma boundary
      cpasma        Plasma current [Amps]   

      F          1D array of f(psi)=R*Bt  [meter-Tesla]
      pres          1D array of p(psi) [Pascals]
      q          1D array of q(psi)
      
      psi           2D array (nx,ny) of poloidal flux
      """
    data = dict.fromkeys(['nx', 'ny', 'rdim', 'zdim', 'rcentr', 'bcentr', 'rleft', 'zmid', 'rmagx', 'zmagx',
                          'simagx', 'sibdry', 'cpasma', 'F', 'pres', 'q', 'psi'])

    grid_1d = equilibrium.coordinates(psi_n=np.linspace(0, 1, nx))

    grid_2d = equilibrium.grid(resolution=(nx, ny), dim="size")

    cocos = equilibrium.cocos

    psi_factor = 1

    if cocos < 10 < cocos_out:
        psi_factor = 2 * np.pi
    elif cocos > 10 > cocos_out:
        psi_factor = 1 / (2 * np.pi)

    # Using center of the computational region as a geometrical center
    # Todo: use center rmin rmax of limiter instead?
    r0 = (equilibrium.R_max + equilibrium.R_min) / 2

    if use_basedata:
        data = basedata_to_dict(equilibrium, cocos_out)
    else:
        data['nx'] = nx
        data['ny'] = ny

        data['rdim'] = equilibrium.R_max - equilibrium.R_min
        data['zdim'] = equilibrium.Z_max - equilibrium.Z_min
        data['rcentr'] = r0
        data['bcentr'] = equilibrium.F0 / r0
        data['rleft'] = equilibrium.R_min
        data['zmid'] = (equilibrium.Z_max + equilibrium.Z_min) / 2

        data['rmagx'] = equilibrium.magnetic_axis.R[0]
        data['zmagx'] = equilibrium.magnetic_axis.Z[0]

        data['simagx'] = equilibrium._psi_axis * psi_factor
        data['sibdry'] = equilibrium._psi_lcfs * psi_factor

        data['cpasma'] = equilibrium.I_plasma

        # Boundary:
        bnd_r = equilibrium.lcfs.R[:-1]
        bnd_z = equilibrium.lcfs.Z[:-1]
        ind = np.argmin(bnd_z).item()

        bnd_r = np.roll(bnd_r, -ind)
        bnd_r = np.append(bnd_r, bnd_r[0])
        bnd_z = np.roll(bnd_z, -ind)
        bnd_z = np.append(bnd_z, bnd_z[0])

        # Downsample plasma boundary
        if nbdry:
            bdry_down_sample = (len(bnd_r) // nbdry) + 1
        else:
            bdry_down_sample = 1

        data["rbdry"] = bnd_r[::bdry_down_sample]
        data["zbdry"] = bnd_z[::bdry_down_sample]

        # 1d profiles:
        data['F'] = equilibrium.F(grid_1d, grid=False)
        data['FFprime'] = equilibrium.FFprime(grid_1d, grid=False) / psi_factor
        data['pres'] = equilibrium.pressure(grid_1d, grid=False)
        data['pprime'] = equilibrium.pprime(grid_1d, grid=False) / psi_factor

        # Check values on axis (!)
        if q_positive:
            data['q'] = equilibrium.abs_q(grid_1d, grid=False)
        else:
            data['q'] = equilibrium.q(grid_1d, grid=False)

        data['psi'] = grid_2d.psi.T * psi_factor

        # Tokamak wall:
        data['rlim'] = equilibrium.first_wall.R
        data['zlim'] = equilibrium.first_wall.Z

    if equilibrium.time_unit == 's':
        time = int(equilibrium.time * 1000)
    else:
        time = int(equilibrium.time)

    with open(file, 'w') as f:
        write_geqdsk(data, f, label, equilibrium.shot, time)
