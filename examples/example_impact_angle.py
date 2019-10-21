# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:23:37 2019

@author: Ondrej
"""
import sys
sys.path.append('C:\\Users\\Ondrej\\Projects\\pleque\\')

import matplotlib.pyplot as plt

from pleque.tests.utils import load_testing_equilibrium
import numpy as np


import matplotlib.pyplot as plt


from pleque.io import _geqdsk as eqdsktool 
from pleque.io.readers import read_geqdsk
from pleque.utils.plotting import *
from pleque import Equilibrium
from pleque.tests.utils import get_test_equilibria_filenames, load_testing_equilibrium
from scipy.interpolate import splprep, splev
#gfiles = get_test_equilibria_filenames()
#eq = read_geqdsk(gfiles[6])

eq = load_testing_equilibrium(4)

surf_lcfs = eq._flux_surface(psi_n=1 - 1e-6)[0]
lim=eq.first_wall
print(dir(lim))

plt.plot(lim.R,lim.Z)
plt.show()

point = eq.coordinates(R=0.83, Z=-0.3)
surf_frompoint = eq._flux_surface(point)
surf_frompoint = eq._flux_surface(point)

grid = eq.grid(resolution=[1e-3, 2e-3], dim="step")
#grint(grid.R)

#Br=eq.B_R(point)
#Bz=eq.B_Z(point)
#Btor=eq.B_tor(point)

#print(Br,Bz,Btor)

def interpolate_lim(lim):
    tck, u = splprep([lim.R, lim.Z], s=0)
    t=np.linspace(np.amin(u),np.amax(u),500)
    newpoints = splev(t, tck)
    return newpoints[0],newpoints[1]

newpoints=interpolate_lim(lim)

def lim_norm_vec_splined(lim):
    dR=-np.diff(lim[0])
    dZ=-np.diff(lim[1])
    lim_vec=np.vstack((dR,dZ,np.zeros(np.shape(dR))))
    print(np.shape(lim_vec))
    pol=lim_vec/np.linalg.norm(lim_vec,axis=0)

    tor=[0,0,1]
    return np.cross(pol,tor,axis=0)

def lim_norm_vec(lim):
    dR=-np.diff(lim.R)
    dZ=-np.diff(lim.Z)
    lim_vec=np.vstack((dR,dZ,np.zeros(np.shape(dR))))
    print(np.shape(lim_vec))
    pol=lim_vec/np.linalg.norm(lim_vec,axis=0)

    tor=[0,0,1]
    return np.cross(pol,tor,axis=0)
    
    

def impact_angle(eq, first_wall):
    """
    Attempt to create firs iteration of an impact angle.
    
    """
    
    normal_vecs=lim_norm_vec_splined(first_wall).T
    fwr=first_wall[0]
    fwz=first_wall[1]
    fwphi=np.zeros(np.shape(fwr))
    first_wall_transp=np.vstack([fwr,fwz,fwphi]).T
    BR=eq.B_R(first_wall_transp)
    
    #ax.plot(fwz,BR)
    Bz=eq.B_Z(first_wall_transp)
    Btor=eq.B_tor(first_wall_transp)
    
    Bvec=np.vstack((BR,Bz,Btor))
    Bvec=Bvec/np.linalg.norm(Bvec,axis=0)
    print(np.shape(Bvec[:,:-1]),np.shape(normal_vecs))
    #impang=np.dot(Bvec[:,:-1],normal_vecs.T,axes=0)
    impcos=np.einsum('ij,ij->j', Bvec[:,:-1], normal_vecs.T,)
    return np.arccos(impcos)
    
normal_vecs=lim_norm_vec_splined(newpoints)

#print(np.shape(normal_vecs))

#plt.plot(normal_vecs[1,:])
#length=0.02
vec=np.linspace(0,0.01,2)

impang=impact_angle(eq,newpoints)

for i in range(0,499):
    plt.plot(newpoints[0][i]+normal_vecs[0][i]*vec,newpoints[1][i]+normal_vecs[1][i]*vec)
    
print(np.shape(impang))

fig,ax=plt.subplots()
ax.plot(impang)
    