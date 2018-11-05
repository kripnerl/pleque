import matplotlib.pyplot as plt
import numpy as np

from pleque.core import Equilibrium
from pleque.io._readgeqdsk import _readeqdsk, readeqdsk_xarray

limiter = np.loadtxt("test_files/compu/limiter_v3_1_iba.dat")
gfile = '/compass/Shared/Exchange/imrisek/MATLAB/COMPASS_U/Scenarios/scenario_1_baseline_eqdsk'

eq = _readeqdsk(gfile)
eq["r_lim"] = limiter[:, 0]
eq["z_lim"] = limiter[:, 1]
eq_xr = readeqdsk_xarray(gfile)
equil = Equilibrium(eq_xr, first_wall=[eq['r_lim'], eq['z_lim']])

r = np.linspace(eq["rleft"], eq["rleft"] + eq["rdim"], eq["nr"])
z = np.linspace(eq["zmid"] - eq["zdim"] / 2, eq["zmid"] + eq["zdim"] / 2, eq["nz"])

figt, ax = plt.subplots()
ax.plot(eq["r_lim"], eq["z_lim"], "-")
ax.plot(eq["r_bound"], eq["z_bound"])
ax.plot(eq["rmagaxis"], eq["zmagaxis"], "x")
ax.contour(r, z, eq["psi"].T)
ax.set_aspect(1)

figxr_2D, ax = plt.subplots()
eq_xr.psi.plot.contour(ax=ax, x="R", y="Z", levels=20)
ax.plot(eq_xr.r_bound.data, eq_xr.z_bound.data)

ax.set_aspect(1)

figxr_1D, ax = plt.subplots()
eq_xr.pressure.plot.line(ax=ax)

plot1d, ax = plt.subplots()
ax.plot(eq["qpsi"], eq["press"])

plt.show()
