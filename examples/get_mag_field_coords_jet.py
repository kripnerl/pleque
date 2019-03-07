"""
This example calculates the 3 components of magnetic field at the poloidal
crossection and retrieves other input parameters for the SOFT code
(synchrotron radiation synthetic diagnostics) - if desired it stores them in a ready input file.

"""


import matplotlib.pyplot as plt
import numpy as np
import h5py

from pleque.core import Equilibrium

from pleque.io.jet import reader_jet

R=np.linspace(1.8,4,55)
Z=np.linspace(-1.9,2.1,100)

R=np.linspace(1.8,4,55)
Z=np.linspace(-1.9,2.1,100)

shot=92459
time=48.1


def B_for_soft(shot,time, R, Z, save=False):
    """
    Main fucniton

    :param shot: JPN
    :param time: time to evaluate the equilibrium
    :param R: radial coordinates of requested B grid
    :param Z: vertical coordinates of requested B grid
    :param save: to be saved in a h5 file or not
    :return:
    """

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

    r = np.vstack([R])
    z = np.vstack([Z])
    print(np.shape(separatrix), np.shape(maxis), np.shape(wall),np.shape(R), np.shape(Z))

    name='JET_{}_t{}s'.format(shot, time)

    desc='test data form non RE shot'

    if save:
        #save to hdf5
        listtosave = [Br, Bphi, Bz, desc, maxis.T, name, r, separatrix, wall, z]

        names = ['Br', 'Bphi', 'Bz', 'desc', 'maxis', 'name', 'r', 'separatrix', 'wall', 'z']
        dtypes = ['f','f','f','s10','f','s10','f','f','f','f']
        savevars=dict(zip(names,listtosave))
        dtypevars=dict(zip(names,dtypes))
        #absolute path here, it is not very nice
        filename_long = '/home/oficke/MAGFIELD_FOR_SOFT/JET_{}_t={}s.h5'.format(shot, time)
        filename = '/home/oficke/MAGFIELD_FOR_SOFT/test{}_{}.h5'.format(shot, time)

        fil=h5py.File(filename, 'w')

        dt=h5py.special_dtype(vlen=str)

        print(maxis,Br)

        for x in savevars.keys():
            print(x)
            if x in ['name','desc']:
                print(np.array(len(savevars[x])), np.shape(np.string_(savevars[x])))
                fil.create_dataset(x, (1,), dtype=dt, data=savevars[x])
            else:
                print(np.shape(savevars[x]))
                fil.create_dataset(x, data=savevars[x])

        fcheck=h5py.File(filename, 'r')
        print(list(fcheck.keys()))
        print(list(fcheck['desc']))


# Calling

B_for_soft(shot, time, R, Z)