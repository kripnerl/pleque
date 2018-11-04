import matplotlib.pyplot as plt
from pleque.io import omas as plomas
from pleque_test.testing_utils import load_testing_equilibrium
import omas
import numpy as np
ods = omas.ODS()



# load the equilibrium data

eq = load_testing_equilibrium()
grid_2d = eq.grid(resolution=[1e-3, 1e-3], dim="step")
grid_1d = eq.coordinates(psi_n=np.linspace(0,1,100))
gridtype = 1


ods = plomas.write(eq)
ods = plomas.write(eq, grid_1d, grid_2d)
ods = plomas.write(eq, grid_1d, grid_2d, gridtype=1, ods=ods)
