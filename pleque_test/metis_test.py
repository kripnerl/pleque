import matplotlib.pyplot as plt
from pleque.core import FluxFuncs, Coordinates
from pleque.io import metis
from pleque_test.testing_utils import load_testing_equilibrium
import pydons
import os
import xarray as xr
import numpy as np

plt.style.use('seaborn-talk')
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

ax[0].set_xlabel(r"$\Psi_{norm}$")
ax[1].set_xlabel(r"$\Psi$")
ax[0].set_ylabel(r"$Te [eV]$")
ax[1].set_ylabel(r"$Te [eV]$")

print(eq.fluxfuncs.keys())

keys = ['pressure', 'jboot', 'n0', 'tip', 'vrot', 'zeff']

n = len(keys)
fig, axs = plt.subplots(n, sharex=True, figsize=(8, 12))

coords = eq.coordinates(psi_n = np.linspace(0,1,30))

for key, ax in zip(keys, axs):
    print(key)
    ax.plot(coords.rho, eq.fluxfuncs[key](coords))
    ax.set_ylabel(key)
ax = axs[-1]
ax.set_xlabel(r'$\rho$')
plt.tight_layout()

def save_it(name, v=1):
    file_dir = '/compass/home/kripner/konference/2018_PhdEvent/fig/'
    plt.savefig(file_dir +  name +'_v' + str(v) + '.png', transparent=True)
    plt.savefig(file_dir +  name +'_v' + str(v) + '.pdf', transparent=True)
save_it('metis', v=1)

plt.show()
