import matplotlib.pyplot as plt
import numpy as np

from pleque.utils.surfaces import find_contour, get_surface, point_in_first_wall, point_inside_curve
from test.testing_utils import load_testing_equilibrium
from pleque.core import Equilibrium, Coordinates

eq = load_testing_equilibrium()

surf_inlcfs = [eq.fluxSurface(level=0.1, inlcfs=True),
               eq.fluxSurface(level=0.5, inlcfs=True),
               eq.fluxSurface(level=0.9, inlcfs=True)]


surf_closed = [eq.fluxSurface(level=0.1, inlcfs=False, closed=True),
               eq.fluxSurface(level=0.5, inlcfs=False, closed=True),
               eq.fluxSurface(level=0.9, inlcfs=False, closed=True)]

surf_opened = [eq.fluxSurface(level=1.05, inlcfs=False, closed=False),
               eq.fluxSurface(level=1.1, inlcfs=False, closed=False),
               eq.fluxSurface(level=1.2, inlcfs=False, closed=False)]

surf_lcfs = eq.fluxSurface(level=1-1e-6)[0]

r, z = eq.get_grid_RZ(base_R=1e-3, base_Z=1e-3, dim="step")
psipol_map = Coordinates(eq, R = r, Z=z, grid=True)


figx, ax = plt.subplots()
cl = ax.contourf(psipol_map.R, psipol_map.Z, psipol_map.psi_n.T, 50)
plt.colorbar(cl)
ax.plot(surf_lcfs.contour[:,0], surf_lcfs.contour[:,1], "-C3")
for i in range(len(surf_inlcfs)):
    for j in range(len(surf_inlcfs[i])):
        ax.plot(surf_inlcfs[i][j].contour[:,0], surf_inlcfs[i][j].contour[:,1], "-C1")

for i in range(len(surf_closed)):
    for j in range(len(surf_closed[i])):
        ax.plot(surf_closed[i][j].contour[:,0], surf_closed[i][j].contour[:,1], "--C4")

for i in range(len(surf_opened)):
    for j in range(len(surf_opened[i])):
        ax.plot(surf_opened[i][j].contour[:,0], surf_opened[i][j].contour[:,1], "--C5")

ax.plot([],[],"-C3",
        label = "lcfs:\n    length = {0:1.2f},\n"
                "    area={1:1.2f},\n    surface={2:1.2f},\n    volume={3:1.2f}".format(surf_lcfs.length,
                                                                               surf_lcfs.area,
                                                                               surf_lcfs.surface,
                                                                               surf_lcfs.volume))
ax.plot([],[], "-C1", label="inside lcfs")
ax.plot([],[], "--C4", label="closed surface")
ax.plot([],[], "--C5", label="opened surface")
ax.set_aspect(1)
ax.legend(loc=(-1.5, 0.5))