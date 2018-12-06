import matplotlib.pyplot as plt
import numpy as np
import h5py

from pleque.core import Equilibrium
from pleque.io.jet import reader_jet

R=np.linspace(1.8,4,110)
Z=np.linspace(-1.9,2.1,200)


def B_for_soft(shot,time, R, Z):

    eq = reader_jet.sal_jet(shot,timex=time)
    B_R=eq.B_R(R, Z)
    B_Z=eq.B_Z(R, Z)
    B_tor=eq.B_tor(R, Z)

    rlim = [np.min(eq.first_wall.R), np.max(eq.first_wall.R)]
    zlim = [np.min(eq.first_wall.Z), np.max(eq.first_wall.Z)]

    size = rlim[1] - rlim[0]
    rlim[0] -= size / 12
    rlim[1] += size / 12

    size = zlim[1] - zlim[0]
    zlim[0] -= size / 12
    zlim[1] += size / 12

    fig, ax=plt.subplots(figsize=(12,5), ncols=3)


    s1=ax[0].pcolormesh(R, Z, B_tor)
    plt.colorbar(s1, ax=ax[0])
    s2=ax[1].pcolormesh(R, Z, B_R)
    plt.colorbar(s2, ax=ax[1])
    s3=ax[2].pcolormesh(R, Z, B_Z)
    plt.colorbar(s3, ax=ax[2])

    for i in range(len(ax)):
        ax[i].set_xlim(rlim)
        ax[i].set_ylim(zlim)

        ax[i].plot(eq.lcfs.R, eq.lcfs.Z, color='k', ls='--', lw=2)
        ax[i].plot(eq.first_wall.R, eq.first_wall.Z, 'k-', lw=2)
        ax[i].set_xlabel('R [m]')
        ax[i].set_ylabel('Z [m]')

        ax[i].set_aspect('equal')

    plt.tight_layout()


    plt.show()

    #check before saving:

    print(np.shape(B_R), np.shape(B_Z), np.shape(B_tor))

    Br=B_R.T
    Bphi=B_tor.T
    Bz=B_Z.T

    separatrix=np.vstack([eq.lcfs.R, eq.lcfs.Z])
    wall = np.vstack([eq.first_wall.R, eq.first_wall.Z])
    maxis=np.vstack([eq.magnetic_axis.R,eq.magnetic_axis.Z])

    print(np.shape(separatrix), np.shape(maxis), np.shape(wall),np.shape(R), np.shape(Z))

    name='JET_{}_t={}s'.format(shot, time)
    desc=''

    #save to hdf5

    #fil=h5py.File('/home/oficke/MAGFIELD_FOR_SOFT/JET_{}_t={}s.h5'.format(shot, time), 'w')
    #for s in



B_for_soft(92400,47.0,R, Z)