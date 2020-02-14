"""Reader for ASDEX Upgrade (AUG) equilibria

Supports the CLISTE outputs EQI and EQH
"""

from scipy.constants import pi, mu_0
import xarray as xr
from .tools import EquilibriaTimeSlices

# Python helper for working with AUG equilibria
# NOTE uses a different dd library than the standard one
import map_equ
# get the one used, because it might have a version string appended :(
dd = getattr(map_equ, map_equ.sf.__module__)

_2PI = 2*pi


def read_aug_first_wall(shot_no, experiment='AUGD', edition=0):
    """Read the R, Z coordinates of the first wall for the given shot

    Uses the YGC shotfile (graphit contours - now actually wolfram)
    from the most recent shot preceding shot_no where YGC was recorded

    WIP: for now just returns all vessel parts points, not just first wall
    """
    most_recent_shot = dd.PreviousShot(shotfile, shot_no, experiment)
    sf = dd.shotfile()          # apparently __init__ was unknown to the authors :(
    if not sf.Open(shotfile, most_recent_shot, experiment, edition):
        raise RuntimeError(f'Cannot open shotfile {experiment}:{shotfile} {shot_no}({edition})')
    R_first_wall = sg.GetSignal('RrGC')
    Z_first_wall = sg.GetSignal('zzGC')
    # to be safe since no __del__ is present :(
    sf.Close()
    return R_first_wall, Z_first_wall


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
    FFprime = eq_aug.get_profile('FFP')    # TODO probably, even though description claims ff'
    Ipol2F_scale = mu_0 / _2PI   # TODO HOTFIX apparently they divided Ipol by mu_0??
    F = Ipol * Ipol2F_scale
    Fprime = dIpol * Ipol2F_scale
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
    }, attrs={
        'cocos': 13,            # for CLISTE according to the COCOS article
    })

    # close AUG shotfile to be safe as it is global in the map_equ module :(
    eq_aug.Close()

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
