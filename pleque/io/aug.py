"""Reader for ASDEX Upgrade (AUG) equilibria

Supports the CLISTE outputs EQI and EQH
"""

from scipy.constants import pi, mu_0
import xarray as xr
from .tools import EquilibriaTimeSlices

# Python helper for working with AUG equilibria
# NOTE uses a different dd library than the standard one
import map_equ

_2PI = 2*pi


def read_aug_eq(shot_no, eq_type='EQH', time=None, experiment='AUGD', edition=0):
    """Read and AUG equilibria from a shotfile


    :param shot_no: discharge number
    :param eq_type: equilibria shotfile, e.g. EQH, EQI
    :param time: time slice [s] to use or if None an EquilibriaTimeSlices is returned
    :param experiment: shotfile category
    :param edition: shotfile edition

    :return Equilibrium or EquilibriaTimeSlices: depending on time given
    """
    eq_aug = map_equ.equ_map(shot_no, eq_type, experiment, edition)
    eq_aug.read_pfm()       # reads the poloidal flux matrix
    eq_aug.read_scalars()   # such as psix=pci_lcfs
    Ipol, dIpol = eq_aug.get_mixed('Jpol')  # poloidal current Ip(psi) and deriv
    ffprime = eq_aug.get_profile('FFP')    # ff'   (f=F * mu_0)
    F = Ipol / _2PI
    Fprime = dIpol / _2PI
    FFprime = ffprime * mu_0**2
    pressure, pprime = eq_aug.get_mixed('Pres')
    q = eq_aug.get_profile('Qpsi')  # q(psi)
    pfl = eq_aug.get_profile('PFL')                # poloidal flux label

    ds = xr.Dataset({
        'psi': (['R', 'Z', 'time'], eq_aug.pfm),  # poloidal flux matrix
        'pressure' : (['time', 'psi_n'], pressure),
        'pprime' : (['time', 'psi_n'], pprime),
        'FFprime' : (['time', 'psi_n'], FFprime),
        'Fprime' : (['time', 'psi_n'], Fprime),
        'F' : (['time', 'psi_n'], F),
        'q' : (['time', 'psi_n'], q),
        'pfl': (['time', 'psi_n'], pfl),  # for consistence
        'psi_lcfs': (['time'], eq_aug.psix),
        'psi_axis': (['time'], eq_aug.psi0),
        # TODO include also other hinting points like mag axis
    }, coords={
        'time': eq_aug.t_eq,
        # NOTE eq_aug.Rmesh is assumed to be constant and probably does not
        # evolve, it typically contains many more time steps than actually used
        'R': eq_aug.Rmesh,
        'Z': eq_aug.Zmesh,
    })
    ds['psi_n_factor'] = ds['psi_lcfs'] - ds['psi_axis']
    ds.coords['psi_nt'] = (ds['pfl'] - ds['psi_axis']) / ds['psi_n_factor']
    # pleque internally uses and expects derivatives with respect to psi_n and
    # will later divide by psi_n_factor, so here it is multiplied in
    ds = ds.update(ds[['pprime', 'Fprime', 'FFprime']] * ds['psi_n_factor'])

    # TODO is it worth doing sortby('time') if only sel is used?
    # Yes, if interpolating
    eqs = EquilibriaTimeSlices(ds, None)

    if time is not None:
        return eqs.get_time_slice(time)
    else:
        return eqs
