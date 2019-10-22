# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:23:37 2019

@author: Ondrej
"""
import sys
sys.path.append('C:\\Users\\Ondrej\\Projects\\pleque\\')

import matplotlib.pyplot as plt

from pleque.tests.utils import load_testing_equilibrium
from pleque.core.coordinates import Coordinates
import numpy as np
from scipy.interpolate import splprep, splev


eq = load_testing_equilibrium(4)

surf_lcfs = eq._flux_surface(psi_n=1 - 1e-6)[0]
first_wall=eq.first_wall


#number of points to resample the limiter
npoints=500



coords=Coordinates(eq,np.vstack((first_wall.R,first_wall.Z)).T)

# Resample using new method
coords2=coords.resample2(npoints)

#just for plotting
newpoints=np.vstack((coords2.R,coords2.Z))


    


# test area

# get normal vector
normal_vecs=coords2.normal_vector().T

# get normalised bvec
bvec=coords2.bvec_at_limiter()

vec=np.linspace(0,0.2,2)

# get impact angle cosinus

impcos=coords2.impact_angle_cos()



fign,axn=plt.subplots()

axn.plot(first_wall.R,first_wall.Z,lw=5,color='c')

axn.set_aspect('equal')
    
newpoints=np.vstack((coords2.R,coords2.Z))
print(np.shape(newpoints))

#plot normal vectors
for i in range(0,npoints-1):
    axn.plot(newpoints[0,i]+normal_vecs[0,i]*vec,newpoints[1,i]+normal_vecs[1,i]*vec,color='k')
    
#plot b vectors
    
for i in range(0,npoints-1):
    axn.plot(newpoints[0,i]+bvec[0,i]*vec,newpoints[1,i]+bvec[1,i]*vec,color='r')
    
print(np.shape(impcos))



fig,ax=plt.subplots()

grid = eq.grid(resolution=[1e-3, 2e-3], dim="step")

cl = axn.contour(grid.R, grid.Z, grid.psi_n, 400, alpha=0.5)

ax.set_aspect('equal')
cmap=plt.get_cmap('jet')

for i in range(0,npoints-1):
    s=ax.scatter(newpoints[0][i], newpoints[1][i], c = cmap((impcos[i]-np.amin(impcos))/(np.amax(impcos)-np.amin(impcos))))

plt.show()
#fig2,ax2=plt.subplots()
#ax2.plot()