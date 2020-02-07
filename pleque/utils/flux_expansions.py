import numpy as np

import pleque


def incidence_angle_sin(coords: 'pleque.Coordinates', vecs):
    """

    :param coords: ``Coordinate`` object (of length ``N_vecs``)of a line in the space on
                  which the incidence angle is evaluated.
    :param vecs: ``array (3, N_vecs)`` vectors in (R, Z, phi) space.
    :return: array of sines of angles of incidence. I.e. cosine of the angle between the normal
             to the line (in the poloidal plane) and the corresponding vector.
    """

    normal_vecs = coords.normal_vector()
    vec_norms = vecs / np.linalg.norm(vecs, axis=0)

    inccos = np.einsum('ij,ij->j', vec_norms, normal_vecs)

    return inccos


def impact_angle_sin(coords: 'pleque.Coordinates'):
    """
    Impact angle calculation - dot product of PFC norm and local magnetic field direction.
    Internally uses `incidence_angle_sin` function where `vecs` are replaced by the vector
    of the magnetic field.

    :return: ``array`` of impact angles cosines
    """

    bvec = coords._eq.Bvec_norm(coords)

    return incidence_angle_sin(coords, bvec)


def impact_angle_cos_pol_projection(coords: 'pleque.Coordinates'):
    """
    Impact angle calculation - dot product of PFC norm and local magnetic field direction
    poloidal projection only.
    Internally uses `incidence_angle_sin` function where `vecs` are replaced by the vector
    of the poloidal magnetic field (Bphi = 0).

    :return: ``array`` of impact angles
    """

    bvec = coords._eq.Bvec(coords)
    bvec[2, :] = 0  # set Z component to be zero

    return incidence_angle_sin(coords, bvec)


def poloidal_mag_flux_exp_coef(equilibrium: 'pleque.Equilibrium', coords: 'pleque.Coordinates'):
    r"""
    **Poloidal magnetic flux expansion coefficient.**


    **Definition:**

    .. math::
       f_\mathrm{pol} = \frac{\Delta r^\mathrm{t}}{\Delta r^\mathrm{u}} =
       \frac{B_\theta^\mathrm{u} R^\mathrm{u}}{B_\theta^\mathrm{t} R^\mathrm{t}}

    **Typical usage:**

    *Poloidal magnetic flux expansion coefficient* is typically used for :math:`\lambda` scaling
    in plane perpendicular to the poloidal component of the magnetic field.

    :param equilibrium: Instance of ``Equilibrium``.
    :param coords: ``Coordinates`` where the coefficient is evaluated.
    :return:
    """

    B_midplane = equilibrium.B_pol(r=coords.r_mid, theta=np.zeros_like(coords.r_mid), grid=False)
    R_mid = equilibrium.magnetic_axis.R[0] + coords.r_mid
    B_coords = equilibrium.B_pol(coords)
    R_coords = coords.R

    return B_midplane * R_mid / (B_coords * R_coords)


def effective_poloidal_mag_flux_exp_coef(equilibrium: 'pleque.Equilibrium', coords: 'pleque.Coordinates'):
    r"""
    **Effective poloidal magnetic flux expansion coefficient**

    **Definition:**

    .. math::
        f_\mathrm{pol, eff} = \frac{B_\theta^\mathrm{u} R^\mathrm{u}}{B_\theta^\mathrm{t} R^\mathrm{t}}
        \frac{1}{\sin \beta} = \frac{f_\mathrm{pol}}{\sin \beta}

    Where :math:`\beta` is inclination angle of the poloidal magnetic field and the target plane.

    **Typical usage:**

    *Effective magnetic flux expansion coefficient* is typically used for :math:`\lambda` scaling
    of the target :math:`\lambda` with respect to the upstream value.

    .. math::
        \lambda^\mathrm{t} = \lambda_q^\mathrm{u} f_{\mathrm{pol, eff}}

    This coefficient can be also used to calculate peak target heat flux from the total power through
    LCFS if the perpendicular diffusion is neglected. Then for the peak value stays

    .. math::
        q_{\perp, \mathrm{peak}} = \frac{P_\mathrm{div}}{2 \pi R^\mathrm{t} \lambda_q^\mathrm{u}}
        \frac{1}{f_\mathrm{pol, eff}}

    Where :math:`P_\mathrm{div}` is total power to outer strike point and $\lambda_q^\mathrm{u}$
    is e-folding length on the outer midplane.

    :param equilibrium: Instance of ``Equilibrium``.
    :param coords: ``Coordinates`` where the coefficient is evaluated.
    :return:
    """

    return poloidal_mag_flux_exp_coef(equilibrium, coords) \
           / np.abs(coords.impact_angle_sin_pol_projection())


def poloidal_heat_flux_exp_coef(equilibrium: 'pleque.Equilibrium', coords: 'pleque.Coordinates'):
    r"""
    **Poloidal heat flux expansion coefficient**

    **Definition:**

    .. math::
        f_\mathrm{pol, heat} = \frac{B_\theta^\mathrm{u}}{B_\theta^\mathrm{t}}

    **Typical usage:**
    *Poloidal heat flux expansion coefficient* is typically used to scale poloidal heat flux
    (heat flux projected along poloidal magnetic field)  along the magnetic field line.

    .. math::
        q_\theta^\mathrm{t} = \frac{q_\theta^\mathrm{u}}{f_{\mathrm{pol, heat}}}


    :param equilibrium: Instance of ``Equilibrium``.
    :param coords: ``Coordinates`` where the coefficient is evaluated.
    :return:
    """

    B_midplane = equilibrium.B_pol(r=coords.r_mid, theta=np.zeros_like(coords.r_mid), grid=False)
    B_coords = equilibrium.B_pol(coords)

    return B_midplane / B_coords


def effective_poloidal_heat_flux_exp_coef(equilibrium: 'pleque.Equilibrium', coords: 'pleque.Coordinates'):
    r"""
    **Effective poloidal heat flux expansion coefficient**

    **Definition:**

    .. math::
        f_\mathrm{pol, heat, eff} = \frac{B_\theta^\mathrm{u}}{B_\theta^\mathrm{t}}
        \frac{1}{\sin \beta} = \frac{f_\mathrm{pol}}{\sin \beta}

    Where :math:`\beta` is inclination angle of the poloidal magnetic field and the target plane.

    **Typical usage:**

    *Effective poloidal heat flux expansion coefficient* is typically used scale upstream poloidal
    heat flux to the target plane.

    .. math::
        q_\perp^\mathrm{t} = \frac{q_\theta^\mathrm{u}}{f_{\mathrm{pol, heat, eff}}}

    :param equilibrium: Instance of ``Equilibrium``.
    :param coords: ``Coordinates`` where the coefficient is evaluated.
    :return:
    """

    return poloidal_heat_flux_exp_coef(equilibrium, coords) \
           / np.abs(coords.impact_angle_sin_pol_projection())


def parallel_heat_flux_exp_coef(equilibrium: 'pleque.Equilibrium', coords: 'pleque.Coordinates'):
    r"""
    **Parallel heat flux expansion coefficient**

    **Definition:**

    .. math::
        f_\parallel= \frac{B^\mathrm{u}}{B^\mathrm{t}}

    **Typical usage:**

    *Parallel heat flux expansion coefficient* is typically used to scale total upstream heat flux
    parallel to the magnetic field along the magnetic field lines.

    .. math::
        q_\parallel^\mathrm{t} = \frac{q_\parallel^\mathrm{u}}{f_\parallel}

    :param equilibrium: Instance of ``Equilibrium``.
    :param coords: ``Coordinates`` where the coefficient is evaluated.
    :return:
    """

    B_midplane = equilibrium.B_abs(r=coords.r_mid, theta=np.zeros_like(coords.r_mid), grid=False)
    B_coords = equilibrium.B_abs(coords)

    return B_midplane / B_coords


def total_heat_flux_exp_coef(equilibrium: 'pleque.Equilibrium', coords: 'pleque.Coordinates'):
    r"""
    **Total heat flux expansion coefficient**

    **Definition:**

    .. math::
        f_\mathrm{tot} = \frac{B^\mathrm{u}}{B^\mathrm{t}} \frac{1}{\sin \alpha} =
        \frac{f_\parallel}{\sin \alpha}

    Where :math:`\alpha` is an inclination angle of the total magnetic field and the target plane.

    .. important:: :math:`\alpha` is an inclination angle of the total magnetic field to the
                   target plate. Whereas :math:`\beta` is an inclination of poloidal components
                   of the magnetic field to the target plate.

    **Typical usage:**

    *Total heat flux expansion coefficient* is typically used to project total upstream heat flux
    parallel to the magnetic field to the target plane.

    .. math::
        q_\perp^\mathrm{t} = \frac{q_\parallel^\mathrm{u}}{f_{\mathrm{tot}}}

    :param equilibrium: Instance of ``Equilibrium``.
    :param coords: ``Coordinates`` where the coefficient is evaluated.
    :return:
    """

    return parallel_heat_flux_exp_coef(equilibrium, coords) \
           / np.abs(coords.impact_angle_sin())
