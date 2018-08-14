import matplotlib.pyplot as plt
from os.path import expanduser
import numpy as np
from pleque.core import Equilibrium
from pleque.io.readgeqdsk import readeqdsk_xarray
from pleque.utils.surfaces import find_contour, get_surface


limiter = np.loadtxt(expanduser("~/Python/modules/pleque/test/test_files/compu/limiter_v3_1_iba.dat"))
gfile = '/compass/Shared/Exchange/imrisek/MATLAB/COMPASS_U/Scenarios/scenario_1_baseline_eqdsk'

eq_fiesta = readeqdsk_xarray(gfile)
eq = Equilibrium(eq_fiesta, first_wall=limiter)
# eq._mg_axis = np.array([0,0.9])

r = np.linspace(eq.r_min, eq.r_max, 300)
z = np.linspace(eq.z_min, eq.z_max, 400)

psipol = eq.psi(R=r, Z=z)
contour = find_contour(psipol, 0.2, r, z)

fluxsurface = get_surface(eq, 0.2, 300, 400)

# isinside =

figx, ax = plt.subplots()
ax.contourf(r, z, psipol.T)
ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], "k")
ax.plot(eq._mg_axis[0], eq._mg_axis[1], "xk")
for i in range(len(contour)):
    ax.plot(contour[i][:, 0], contour[i][:, 1], "r")
for i in range(len(fluxsurface)):
    ax.plot(fluxsurface[i][:, 0], fluxsurface[i][:, 1], "m")
ax.set_aspect(1)