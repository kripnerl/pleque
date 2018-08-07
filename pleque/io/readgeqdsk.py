import numpy as np
import re
import xarray as xr

def readeqdsk_xarray(filepath):
    #TODO: Put the dictionary into a xarray for better user experience
    #TODO: I need to find out how to calculate r coordinates correctly
    pass

def readeqdsk(filepath):
    """
    Read equidisk file into a python dictionary.
    :param filepath: path to the eqdsk file
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

    #dig out the variables
    equi["rdim"] = equistuff[0]
    equi["zdim"] = equistuff[1]
    equi["rcentr"] = equistuff[2]
    equi["rleft"] = equistuff[3]
    equi["zmid"] = equistuff[4]
    equi["rmagaxis"] = equistuff[5]
    equi["zmagaxis"] = equistuff[6]
    equi["psimagaxis"] = equistuff[7]
    equi["psibdry"] = equistuff[8]
    equi["bcentr"] = equistuff[9]
    equi["cpasma"] = equistuff[10]
    equi["psimagaxis"] = equistuff[11]
    equi["rmagaxis"] = equistuff[13]
    equi["zmagaxis"] = equistuff[15]
    equi["psibdry"] = equistuff[17]


    #get the 1d and 2d arrays
    names = ["fpol", "press", "ffprime", "pprime", "psi", "qpsi"]
    index = 20

    for i in names:
        if i == "psi": #only psi is 2D array
            equi[i] = np.reshape(equistuff[index: index + equi["nr"] * equi["nz"]],
                                 newshape=(equi["nr"], equi["nz"]))
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