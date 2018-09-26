import matplotlib.pyplot as plt
from pleque.core import FluxFuncs, Coordinates
from pleque.io import metis
from test.testing_utils import load_testing_equilibrium
import pydons
import os
import xarray as xr
import numpy as np


file = "/compass/Shared/Common/IT/projects/compass-u-vp3/metis-scenarios/scans/2018-09-25/CU_Z12n25_Bt40Ip12_EC4.mat"
time = 1


eq = load_testing_equilibrium()
#read saved metis simulation into fluxfunction
fluxfun = metis.read(eq, file, time)

#creace coordinates to use for plotting
coords = eq.coordinates(psi_n = np.linspace(0,1,30))

figte, ax = plt.subplots(1,2)
ax[0].plot(coords.psi_n, fluxfun.tep(coords))
ax[1].plot(coords.psi, fluxfun["tep"](coords))

ax[0].set_xlabel("$\Psi_{norm}$")
ax[1].set_xlabel("$\Psi$")
ax[0].set_ylabel("$Te [eV]")
ax[1].set_ylabel("$Te [eV]")
