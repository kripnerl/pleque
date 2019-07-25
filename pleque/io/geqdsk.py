from ._geqdsk import read_as_equilibrium
from ._geqdsk import write as write_geqdsk
import numpy as np
import pleque


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


def write(equilibrium: pleque.Equilibrium, file, nx=64, ny=128, label=None, cocos_out=3):
    """
    Write a GEQDSK equilibrium file.

    :param equilibrium: pleque.Equilibrum
    :param file: str, file name
    :param nx: int, R-dimension
    :param ny: int, Z-dimension
    :param label: str, max 11 characters long text added on the beginning of g-file (default is PLEQUE)
    :param cocos_out: not used at the moment
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

    # Using center of the computational region as a geometrical center
    # Todo: use center rmin rmax of limiter instead?
    r0 = (equilibrium.R_max + equilibrium.R_min) / 2

    data['nx'] = nx
    data['ny'] = ny
    # XXX
    data['rdim'] = equilibrium.R_max - equilibrium.R_min
    data['zdim'] = equilibrium.Z_max - equilibrium.Z_min
    data['rcentr'] = r0
    data['bcentr'] = equilibrium.F0 / r0
    data['rleft'] = equilibrium.R_min
    data['zmid'] = (equilibrium.Z_max + equilibrium.Z_min) / 2

    data['rmagx'] = equilibrium.magnetic_axis.R[0]
    data['zmagx'] = equilibrium.magnetic_axis.Z[0]
    data['simagx'] = equilibrium._psi_axis

    data['sibdry'] = equilibrium._psi_lcfs
    data['cpasma'] = equilibrium.I_plasma

    # Boundary:
    bnd_r = equilibrium.lcfs.R[:-1]
    bnd_z = equilibrium.lcfs.Z[:-1]
    ind = np.argmin(bnd_z)

    bnd_r = np.roll(bnd_r, -ind)
    bnd_r = np.append(bnd_r, bnd_r[0])
    bnd_z = np.roll(bnd_z, -ind)
    bnd_z = np.append(bnd_z, bnd_z[0])

    # TODO Something more clever...
    # XXX force downsample:
    data["rbdry"] = bnd_r[::12]
    data["zbdry"] = bnd_z[::12]

    # 1d profiles:
    data['F'] = equilibrium.F(grid_1d, grid=False)
    data['FFprime'] = equilibrium.FFprime(grid_1d, grid=False)
    data['pres'] = equilibrium.pressure(grid_1d, grid=False)
    data['pprime'] = equilibrium.pprime(grid_1d, grid=False)
    # Check values on axis (!)
    data['q'] = equilibrium.q(grid_1d, grid=False)

    data['psi'] = grid_2d.psi.T

    # Tokamak wall:
    data['rlim'] = equilibrium.first_wall.R
    data['zlim'] = equilibrium.first_wall.Z

    if equilibrium.time_unit == 's':
        time = int(equilibrium.time * 1000)
    else:
        time = int(equilibrium.time)

    with open(file, 'w') as f:
        write_geqdsk(data, f, label, equilibrium.shot, time)
