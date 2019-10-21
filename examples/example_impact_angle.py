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


npoints=500

fign,axn=plt.subplots()

axn.plot(first_wall.R,first_wall.Z)


grid = eq.grid(resolution=[1e-3, 2e-3], dim="step")

coords=Coordinates(eq,np.vstack((first_wall.R,first_wall.Z)).T)


def interpolate_lim(coords, npoints):
    """
    Implicit spline curve interpolation for the limiter, number of points must be specified

    """
    
    dR=-np.diff(first_wall.R)
    dZ=-np.diff(first_wall.Z)
    
    
    dists=coords.cum_length
    
    #dists=np.cumsum(np.sqrt(dR**2+dZ**2))
    #print(np.shape(first_wall.R[:-1]),np.shape(dists))
    #print(dists)
    
    tck, u = splprep([coords.R, first_wall.Z],u=dists,k=1,s=0)
    t=np.linspace(np.amin(u),np.amax(u),npoints)
    newpoints = splev(t, tck)
    return newpoints[0],newpoints[1]



def limiter_norm_vec_splined(first_wall):
    """Calculate limiter fw input as list of R and Z cooridnate arrays
    
    """
    dR=-np.diff(first_wall[0])
    dZ=-np.diff(first_wall[1])
    lim_vec=np.vstack((dR,dZ,np.zeros(np.shape(dR))))
    print(np.shape(lim_vec))
    pol=lim_vec/np.linalg.norm(lim_vec,axis=0)

    tor=[0,0,1]
    
    normal=np.cross(pol,tor,axis=0)/np.linalg.norm(np.cross(pol,tor,axis=0))
    
    return normal

def limimiter_norm_vec(first_wall):
    """Calculate limiter nomrmal vector with fw input directly from eq class
    
    """
    dR=-np.diff(first_wall.R)
    dZ=-np.diff(first_wall.Z)
    lim_vec=np.vstack((dR,dZ,np.zeros(np.shape(dR))))
    print(np.shape(lim_vec))
    pol=lim_vec/np.linalg.norm(lim_vec,axis=0)

    tor=[0,0,1]
    
    normal=np.cross(pol,tor,axis=0)/np.linalg.norm(np.cross(pol,tor,axis=0))
    
    return normal
    
def bvec_at_limiter(eq, first_wall):
    """Impact angle calculation - dot product of PFC norm and local magnetic field direction
    
    Parameters
    ----------
    eq: object equilibrium
    firstwall: interpolated first wall
        
    Returns
    -------
    Bvec: magnetic field direction at the limiter
        
    Note
    ----
    
    
    """
    fwr=first_wall[0]
    fwz=first_wall[1]
    fwphi=np.zeros(np.shape(fwr))
    
    first_wall_transp=np.vstack([fwr,fwz,fwphi]).T
    
    
    bR=eq.B_R(first_wall_transp)
    bz=eq.B_Z(first_wall_transp)
    btor=eq.B_tor(first_wall_transp)
    
    bvec=np.vstack((bR,bz,btor))
    bvec=bvec/np.linalg.norm(bvec,axis=0)
    
    return bvec
   

def impact_angle(eq, first_wall):
    """Impact angle calculation - dot product of PFC norm and local magnetic field direction
    
    Parameters
    ----------
    eq: object equilibrium
    firstwall: interpolated first wall
        
    Returns
    -------
    impang: impact angle in rad
        
    Note
    ----
    
    
    """
    
    normal_vecs=limiter_norm_vec_splined(first_wall).T
    
    bvec=bvec_at_limiter(eq,first_wall)
    #print(np.shape(Bvec[:,:-1]),np.shape(normal_vecs))
    impcos=np.einsum('ij,ij->j', bvec[:,:-1], normal_vecs.T)
    
    return np.arccos(impcos)
    


# test area

newpoints=interpolate_lim(first_wall,npoints)

normal_vecs=limiter_norm_vec_splined(newpoints)

bvec=bvec_at_limiter(eq, newpoints)

vec=np.linspace(0,0.1,2)

impang=impact_angle(eq,newpoints)

axn.set_aspect('equal')
    
for i in range(0,npoints-1):
    axn.plot(newpoints[0][i]+normal_vecs[0][i]*vec,newpoints[1][i]+normal_vecs[1][i]*vec,color='k')
    
#figb,axb=plt.subplots()
    
for i in range(0,npoints-1):
    axn.plot(newpoints[0][i]+bvec[0][i]*vec,newpoints[1][i]+bvec[1][i]*vec,color='r')
    
print(np.shape(impang))

fig,ax=plt.subplots()

grid = eq.grid(resolution=[1e-3, 2e-3], dim="step")
# psipol_map = Coordinates(eq, R = r, Z=z, grid=True)
cl = axn.contour(grid.R, grid.Z, grid.psi_n, 200)

ax.set_aspect('equal')
cmap=plt.get_cmap('jet')
for i in range(0,npoints-1):
    ax.scatter(newpoints[0][i], newpoints[1][i], c = cmap((impang[i]-np.amin(impang))/(np.amax(impang)-np.amin(impang))))
    
#fig2,ax2=plt.subplots()
#ax2.plot()