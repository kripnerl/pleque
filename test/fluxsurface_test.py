import matplotlib.pyplot as plt

from test.testing_utils import load_testing_equilibrium

eq = load_testing_equilibrium()

surf_inlcfs = [eq.fluxSurface(eq.coordinates(psi_n = 0.1), inlcfs=True),
               eq.fluxSurface(eq.coordinates(psi_n = 0.5), inlcfs=True),
               eq.fluxSurface(eq.coordinates(psi_n = 0.9), inlcfs=True)]


surf_closed = [eq.fluxSurface(eq.coordinates(psi_n = 0.1), inlcfs=False, closed=True),
               eq.fluxSurface(eq.coordinates(psi_n = 0.5), inlcfs=False, closed=True),
               eq.fluxSurface(eq.coordinates(psi_n = 0.9), inlcfs=False, closed=True)]

surf_opened = [eq.fluxSurface(eq.coordinates(psi_n = 0.1), inlcfs=False, closed=False),
               eq.fluxSurface(eq.coordinates(psi_n = 0.5), inlcfs=False, closed=False),
               eq.fluxSurface(eq.coordinates(psi_n = 0.9), inlcfs=False, closed=False)]

surf_lcfs = eq.fluxSurface(psi_n=1-1e-6)[0]

point = eq.coordinates(R=0.83, Z = -0.3)
surf_frompoint = eq.fluxSurface(point)


grid = eq.get_grid_RZ(resolution=[1e-3, 2e-3], dim="step")
#psipol_map = Coordinates(eq, R = r, Z=z, grid=True)


figx, ax = plt.subplots()
cl = ax.contourf(grid.R, grid.Z, grid.psi_n.T, 50)
plt.colorbar(cl)
ax.plot(surf_lcfs.contour.as_array()[:, 0], surf_lcfs.contour.as_array()[:, 1], "-C3")
for i in range(len(surf_inlcfs)):
    for j in range(len(surf_inlcfs[i])):
        ax.plot(surf_inlcfs[i][j].contour.as_array()[:, 0], surf_inlcfs[i][j].contour.as_array()[:, 1], "-C1")

for i in range(len(surf_closed)):
    for j in range(len(surf_closed[i])):
        ax.plot(surf_closed[i][j].contour.as_array()[:, 0], surf_closed[i][j].contour.as_array()[:, 1], "--C4")

for i in range(len(surf_opened)):
    for j in range(len(surf_opened[i])):
        ax.plot(surf_opened[i][j].contour.as_array()[:, 0], surf_opened[i][j].contour.as_array()[:, 1], "--C5")
ax.plot(surf_frompoint[0].contour.as_array()[:, 0], surf_frompoint[0].contour.as_array()[:, 1], "--C6",
        label="surface throught point")
ax.plot(point.R, point.Z, "xC6")
ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], "k")

ax.plot([], [], "-C3",
        label="lcfs:\n    length = {0:1.2f},\n"
              "    area={1:1.2f},\n    surface={2:1.2f},\n    volume={3:1.2f}".format(surf_lcfs.length,
                                                                                      surf_lcfs.area,
                                                                                      surf_lcfs.surface,
                                                                                      surf_lcfs.volume))
ax.plot([], [], "-C1", label="inside lcfs")
ax.plot([], [], "--C4", label="closed surface")
ax.plot([], [], "--C5", label="opened surface")
ax.set_aspect(1)
ax.legend(loc=(-1.5, 0.5))
