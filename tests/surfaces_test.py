import matplotlib.pyplot as plt
import numpy as np

from pleque.utils.surfaces import find_contour, get_surface, point_in_first_wall, point_inside_curve
from pleque.tests.utils import load_testing_equilibrium

eq = load_testing_equilibrium()
# eq._mg_axis = np.array([0,0.9])

r = np.linspace(eq.R_min, eq.R_max, 300)
z = np.linspace(eq.Z_min, eq.Z_max, 400)

psipol = eq.psi(R=r, Z=z)
contour = find_contour(psipol, 0.2, r, z)

fluxsurface = get_surface(eq, eq._psi_lcfs, norm=False)
print(r'$\psi$ (lcfs) = {:0.3f}'.format(eq._psi_lcfs))

mesh_r, mesh_z = np.meshgrid(r[::10], z[::10])
points = np.vstack((mesh_r.ravel(), mesh_z.ravel())).T

inside_fw = point_in_first_wall(eq, points)
inside_lcfs = point_inside_curve(points, eq._lcfs)
mask_fw = inside_fw.reshape(mesh_z.shape)
mask_lcfs = inside_lcfs.reshape(mesh_z.shape)

figx, ax = plt.subplots()
cl = ax.contourf(r, z, psipol)
plt.colorbar(cl)
ax.plot(eq._first_wall[:, 0], eq._first_wall[:, 1], "k")
ax.plot(eq._mg_axis[0], eq._mg_axis[1], "xk")
for i in range(len(contour)):
    ax.plot(contour[i][:, 0], contour[i][:, 1], "r")
for i in range(len(fluxsurface)):
    ax.plot(fluxsurface[i][:, 0], fluxsurface[i][:, 1], color='white')
ax.plot(eq._lcfs[:, 0], eq._lcfs[:, 1], '--m')

ax.plot(mesh_r[mask_fw], mesh_z[mask_fw], '.g', alpha=0.3)
ax.plot(mesh_r[mask_lcfs], mesh_z[mask_lcfs], '.m', alpha=0.3)
ax.plot(mesh_r[~mask_fw], mesh_z[~mask_fw], '.r', alpha=0.3)
ax.set_aspect(1)

plt.show()
