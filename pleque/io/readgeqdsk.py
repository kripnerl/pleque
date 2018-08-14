import numpy as np
import re
import xarray as xr

def readeqdsk_xarray(filepath, order = "F"):
    """
    Read the eqdsk file contents into an xarray
    :param filepath: path to the gfile
    :param order: dimension convention of psi poloidal 2D array ("C", "F", ...), default is "F"
    :return: xarray
    """
    eq = readeqdsk(filepath, order = order)

    #calculate r, z coordinates for 2d psi profile
    r_psi = np.linspace(eq["rleft"], eq["rleft"] + eq["rdim"], eq["nr"])
    z_psi = np.linspace(eq["zmid"] - eq["zdim"]/2, eq["zmid"] + eq["zdim"]/2, eq["nz"])


    eq_xarray = xr.Dataset({"psi":(("r_psi", "z_psi"), eq["psi"]),  #2d psi poloidal profile
                            "r_bound":eq["r_bound"], "z_bound":eq["z_bound"], #plasma boundary
                            "r_lim": eq["r_lim"], "z_lim": eq["r_lim"],
                            "fpol": ("qpsi",eq["fpol"]),
                            "press": ("qpsi",eq["press"]),
                            "ffprime": ("qpsi",eq["ffprime"]),
                            "pprime": ("qpsi",eq["pprime"])}, #limiter contour
                           coords={"r_psi":r_psi,
                                   "z_psi":z_psi,
                                   "qpsi": eq["qpsi"]})

    attrs = ["rdim", "zdim", "rcentr", "rleft", "zmid", "rmagaxis", "zmagaxis", "psimagaxis",
             "psibdry", "bcentr", "cpasma", "psimagaxis","rmagaxis", "zmagaxis",  "psibdry"]

    for i in attrs:
        eq_xarray.attrs[i] = eq[i]


    return eq_xarray

def readeqdsk(filepath, order = "F"):
    """
    Read equidisk file into a python dictionary.
    :param filepath: path to the eqdsk file
    :param order: dimension convention of psi poloidal 2D array ("C", "F", ...), default is "F"
    :return: dictionary with equilibrium data
    """

    #read the file
    with open(filepath,"r") as f:
        data = f.read()

    pattern_floats = re.compile("[+-]{1}[0-9]+\.\d+e{1}[+-]{1}\d+") #regexp for float trains
    pattern_headers = re.compile(" +.+\n{1}") #regexp for headers which contain spaces

    headers = list(re.finditer(pattern_headers, data))

    equi = {} #dict with reconstruction data

    #get the usefull info from the first header
    header = headers[0].group()
    header = re.sub("\n","", header)
    header = re.split(" +",header)

    equi["nr"] = int(header[-2])
    equi["nz"] = int(header[-1])

    #get the first block of number from in-between the two first headers
    equistuff = data[headers[0].span()[1]: headers[1].span()[0]]
    #be it a numpy array!
    equistuff = re.findall(pattern_floats, equistuff)
    equistuff = np.asarray(equistuff).astype(float)

    # attribute names and positions
    attrs = ["rdim", "zdim", "rcentr", "rleft", "zmid", "rmagaxis", "zmagaxis", "psimagaxis",
             "psibdry", "bcentr", "cpasma", "psimagaxis","rmagaxis", "zmagaxis",  "psibdry"]
    attrs_pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 11, 13, 15, 17]

    # get the attributes
    for i in range(len(attrs)):
        equi[attrs[i]] = equistuff[attrs_pos[i]]

    #get the 1d and 2d arrays
    names = ["fpol", "press", "ffprime", "pprime", "psi", "qpsi"]
    index = 20

    for i in names:
        if i == "psi": #only psi is 2D array
            equi[i] = np.reshape(equistuff[index: index + equi["nr"] * equi["nz"]],
                                 newshape=(equi["nr"], equi["nz"]), order = order) # [r, z]
            index += equi["nr"] * equi["nz"]
        else:#read 1d arrays
            equi[i] = equistuff[index:index + equi["nr"]]
            index += equi["nr"]

    # now the boundary points

    header = headers[1].group()
    header = re.sub("\n","", header)
    header = re.split(" +",header)

    equi["nbound"] = int(header[-2])
    equi["nlim"] = int(header[-1])

    # get teh boundary points
    boundage = data[headers[1].span()[1]:headers[2].span()[0]] #the variable name is not a typo!
    boundage = re.findall(pattern_floats, boundage)
    boundage = np.asarray(boundage).astype(float)

    #shape it into 2d array of appropriate shape
    boundage = np.reshape(boundage, newshape = (equi["nbound"] + equi["nlim"], 2))

    equi["r_bound"] = boundage[0:equi["nbound"], 0]
    equi["z_bound"] = boundage[0:equi["nbound"], 1]

    equi["r_lim"] = boundage[equi["nbound"]:equi["nlim"], 0]
    equi["z_lim"] = boundage[equi["nbound"]:equi["nlim"], 1]

    return equi
