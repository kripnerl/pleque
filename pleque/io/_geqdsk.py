"""
Low level routines for reading and writing G-EQDSK files

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.

"""

from datetime import date

from numpy import zeros, pi

from ._fileutils import f2s, ChunkOutput, write_1d, write_2d, next_value

import numpy as np
import xarray as xa
from pleque.core import Equilibrium

def write(data, fh, label=None, shot=None, time=None):
    """
    Write a GEQDSK equilibrium file, given a dictionary of data
    
    data - dictionary
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
    
    fh - file handle
    
    label - Text label to put in the file
    """

    nx = data["nx"]
    ny = data["ny"]

    if not label:
        label = "PLEQUE"
    if len(label) > 11:
        label = label[0:12]
        print('WARNING: label too long, it will be shortened to {}'.format(label))

    creation_date = date.today().strftime("%d/%m/%Y")

    if not shot:
        shot = 0

    if isinstance(shot, int):
        shot = '# {:d}'.format(shot)

    if not time:
        time = 0

    if isinstance(time, int):
        time = '  {:d}ms'.format(time)

    # I have no idea what idum is, here it is set to 3
    idum = 3
    header = "{0:11s}{1:10s}   {2:>8s}{3:16s}{4:4d}{5:4d}{6:4d}\n" \
        .format(label, creation_date, shot, time, idum, nx, ny)

    # First line: Identification string, followed by resolution
    fh.write(header)

    # Second line
    fh.write(
        f2s(data["rdim"]) + f2s(data["zdim"]) + f2s(data["rcentr"]) + f2s(data["rleft"]) + f2s(data["zmid"]) + "\n")

    # Third line
    fh.write(f2s(data["rmagx"]) + f2s(data["zmagx"]) + f2s(data["simagx"]) + f2s(data["sibdry"]) + f2s(
        data["bcentr"]) + "\n")

    # 4th line
    fh.write(f2s(data["cpasma"]) + f2s(data["simagx"]) + f2s(0.0) + f2s(data["rmagx"]) + f2s(0.0) + "\n")

    # 5th line
    fh.write(f2s(data["zmagx"]) + f2s(0.0) + f2s(data["sibdry"]) + f2s(0.0) + f2s(0.0) + "\n")

    # fill arrays
    workk = zeros([nx])

    # Write arrays
    co = ChunkOutput(fh)

    write_1d(data["F"], co)
    write_1d(data["pres"], co)
    if 'FFprime' in data:
        write_1d(data["FFprime"], co)
    else:
        write_1d(workk, co)
    if 'pprime' in data:
        write_1d(data["pprime"], co)
    else:
        write_1d(workk, co)

    write_2d(data["psi"], co)
    write_1d(data["q"], co)

    # Boundary / limiters

    nbdry = 0
    nlim = 0
    if "rbdry" in data:
        nbdry = len(data["rbdry"])
    if "rlim" in data:
        nlim = len(data["rlim"])

    co.newline()
    fh.write("{0:5d}{1:5d}\n".format(nbdry, nlim))

    if nbdry > 0:
        for r, z in zip(data["rbdry"], data["zbdry"]):
            co.write(r)
            co.write(z)
        co.newline()

    if nlim > 0:
        for r, z in zip(data["rlim"], data["zlim"]):
            co.write(r)
            co.write(z)
        co.newline()


def read(fh, cocos=1):
    """
    Read a G-EQDSK formatted equilibrium file
    
    Format is specified here:
    https://fusion.gat.com/theory/Efitgeqdsk

    cocos   - COordinate COnventions. Not fully handled yet,
              only whether psi is divided by 2pi or not.
              if < 10 then psi is divided by 2pi, otherwise not.

    Returns
    -------
    
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

    # Read the first line
    header = fh.readline()
    words = header.split()  # Split on whitespace
    if len(words) < 3:
        raise ValueError("Expecting at least 3 numbers on first line")

    idum = int(words[-3])
    nx = int(words[-2])
    ny = int(words[-1])

    print("  nx = {0}, ny = {1}".format(nx, ny))

    # Dictionary to hold result
    data = {"nx": nx, "ny": ny}

    # List of fields to read. None means discard value
    fields = ["rdim", "zdim", "rcentr", "rleft", "zmid",
              "rmagx", "zmagx", "simagx", "sibdry", "bcentr",
              "cpasma", "simagx", None, "rmagx", None,
              "zmagx", None, "sibdry", None, None]

    values = next_value(fh)

    for f in fields:
        val = next(values)
        if f:
            data[f] = val

    # Read arrays

    def read_1d(n):
        """
        Read a 1D array of length n
        """
        val = zeros(n)
        for i in range(n):
            val[i] = next(values)
        return val

    def read_2d(n, m):
        """
        Read a 2D (n,m) array in Fortran order
        """
        val = zeros([n, m])
        for y in range(m):
            for x in range(n):
                val[x, y] = next(values)
        return val

    data["F"] = read_1d(nx)
    data["pres"] = read_1d(nx)
    data["FFprime"] = read_1d(nx)
    data["pprime"] = read_1d(nx)

    data["psi"] = read_2d(nx, ny)

    data["q"] = read_1d(nx)

    # Ensure that psi is divided by 2pi
    if cocos > 10:
        for var in ["psi", "simagx", "sibdry"]:
            data[var] /= 2 * pi

    nbdry = next(values)
    nlim = next(values)

    print(nbdry, nlim)

    if nbdry > 0:
        # Read (R,Z) pairs
        data["rbdry"] = zeros(nbdry)
        data["zbdry"] = zeros(nbdry)
        for i in range(nbdry):
            data["rbdry"][i] = next(values)
            data["zbdry"][i] = next(values)

    if nlim > 0:
        # Read (R,Z) pairs
        data["rlim"] = zeros(nlim)
        data["zlim"] = zeros(nlim)
        for i in range(nlim):
            data["rlim"][i] = next(values)
            data["zlim"][i] = next(values)

    return data


def data_as_ds(data):
    """
    Convert data mappable to Pandas

    :param data:
    :return:
    """

    r_axis = np.linspace(data["rleft"], data["rleft"] + data["rdim"], data["nx"])
    z_axis = np.linspace(data["zmid"] - data["zdim"] / 2, data["zmid"] + data["zdim"] / 2, data["ny"])
    psi_n = np.linspace(0, 1, len(data['q']))

    eq_xarray = xa.Dataset({"psi": (("R", "Z"), data["psi"]),  # 2d psi poloidal profile
                            "r_bound": data["rbdry"], "z_bound": data["zbdry"],  # plasma boundary
                            "r_lim": data["rlim"], "z_lim": data["zlim"],
                            "F": ("psi_n", data["F"]),
                            "pressure": ("psi_n", data["pres"]),
                            "FFprime": ("psi_n", data["FFprime"]),
                            "q": ("psi_n", data["q"]),
                            "pprime": ("psi_n", data["pprime"])},  # limiter contour
                           coords={"R": r_axis,
                                   "Z": z_axis,
                                   "psi_n": psi_n})

    attrs = ["rdim", "zdim", "rcentr", "rleft", "zmid", "rmagx", "zmagx", "simagx",
             "sibdry", "bcentr", "cpasma", "simagx", "rmagx", "zmagx", "sibdry"]

    for i in attrs:
        eq_xarray.attrs[i] = data[i]
    return eq_xarray


def read_as_equilibrium(fh, cocos=1):
    """
    Read the eqdsk file and open it as `Equilibrium`.

    :param fh: file handler
    :param cocos:
    :return: instance of `Equilibrium`
    """

    data = read(fh, cocos)
    ds = data_as_ds(data)  # as dataset
    fw = np.stack((ds['r_lim'].data, ds['z_lim'].data)).T  # first wall
    eq = Equilibrium(ds, fw)
    return eq
