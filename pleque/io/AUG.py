"""Reader for ASDEX Upgrade (AUG) equilibria

Supports the CLISTE outputs EQI and EQH
"""

import numpy as np
import xarray as xr
from .tools import EquilibriaTimeSlices

import dd                       # the AUG data reading library

def read_aug_eq(shot_no, eq_type='EQH', time=None, experiment='AUGD', edition=0):
    """Read and AUG equilibria from a shotfile


    :param shot_no: discharge number
    :param eq_type: equilibria shotfile, e.g. EQH, EQI
    :param time: time slice [s] to use or if None an EquilibriaTimeSlices is returned
    :param experiment: shotfile category
    :param edition: shotfile edition

    :return Equilibrium or EquilibriaTimeSlices: depending on time given
    """
    shotfile = dd.shotfile(shot, eq_type, experiment, edition)
    pres = shotfile('Pres')     # defined with respect to pfl
    pfl = pres.area.data # poloidal flux label (psi)  TODO is the area safe?
    time = pres.time.data
    pressure = pres.data[::2]        # every second value is pprime
    pprime = pres.data[1::2]
    ffprime = shotfile('FFP').data  # TODO small F or f?
    q = shotfile('Qpsi').data
    psi_RZ = shotfile('PFM')  # Poloidal Flux Matrix; index1=i(0:M);index2=j(0:N); ind3= time
    R = shotfile('Ri')
    Z = shotfile('Zj')
    # the *xx data have index: mag. axis, separatrix, limiter, 2nd sep., 2nd. limiter
    pfxx = shotfile('PFxx')     # psi at interesting points
    # TODO RPFx and zPFx might also be useful as starting points
    psi_axis = pfxx.data[0]
    psi_lcfs = pfxx.data[1]
    # TODO should pleque be given psi and define psi_n after LCFS and axis are found for consistence?
    psi_n_factor = (psi_lcfs - psi_axis)    # factor for psi_n derivatives
    psi_n = (pfl - psi_axis) / psi_n_factor  # TODO should be a function

    ds = xr.Dataset({
        'psi': (['R', 'Z', 'time'], psi_RZ),  # poloidal flux matrix
        'pressure' : (['psi_n', 'time'], pressure),
        # pleque internally uses derivatives with respect to psi_n and will later divide by this factor
        'pprime' : (['psi_n', 'time'], pprime * psi_n_factor),
        'FFprime' : (['psi_n', 'time'], ffprime * psi_n_factor),
        'q' : (['psi_n', 'time'], q),
        'pfl': (['psi_n'], pfl),  # for consistence
    }, coords={
        'time': time,
        'Rt': (['R', 'time'], R),
        'Zt': (['Z', 'time'], Z),
        'psi_n': psi_n,
    })


    # TODO is it worth doing sortby('time') if only sel is used?
    eqs = EquilibriaTimeSlices(ds)

    if time is not None:
        return eqs.get_time_slice(time)
    else:
        return eqs
