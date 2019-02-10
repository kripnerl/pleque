import matplotlib.pyplot as plt
import numpy as np
import pkg_resources

from pleque.io import metis
from pleque.tests.utils import load_testing_equilibrium

plt.style.use('seaborn-talk')
file = pkg_resources.resource_filename("pleque", "resources/metis.mat")
time = 1

eq = load_testing_equilibrium()
# read saved metis simulation into fluxfunction
fluxfun = metis.read(eq, file, time)

# creace coordinates to use for plotting
coords = eq.coordinates(psi_n=np.linspace(0, 1, 30))

figte, ax = plt.subplots(1, 2)
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

coords = eq.coordinates(psi_n=np.linspace(0, 1, 30))

for key, ax in zip(keys, axs):
    print(key)
    ax.plot(coords.rho, eq.fluxfuncs[key](coords))
    ax.set_ylabel(key)
ax = axs[-1]
ax.set_xlabel(r'$\rho$')
plt.tight_layout()
