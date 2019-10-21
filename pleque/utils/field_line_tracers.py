import numpy as np

from pleque import Equilibrium


def _trace_field_line_first_attempt(eq: Equilibrium, *coordinates, coord_type=None, sign=1, step=5e-1, t_max=150,
                                    **coords):
    """Deprecated.
    :param coordinates:
    :param coord_type:
    :param coords:
    :return:
    """
    # from numba import autojit
    from scipy.integrate import ode
    crds = eq.coordinates(*coordinates, coord_type=coord_type, grid=False, **coords)

    def b_norm(t, y):
        # todo: as soon as it will be implemented properly implement as
        # crd = eq.coordinates(y[0], y[1], grid=False)
        # ... it should bring significant acceleration
        B = [eq.B_R(R=y[0], Z=y[1], grid=False),
             eq.B_Z(R=y[0], Z=y[1], grid=False),
             eq.B_tor(R=y[0], Z=y[1], grid=False) / y[0]]
        B = sign * B / np.linalg.norm(B)
        return B

    def do_step(r, z):
        times = []
        pos = []
        y0 = [r, z, 0]
        t0 = 0.
        pos.append(y0)
        times.append(t0)

        integrator = ode(b_norm).set_integrator('dopri5')
        integrator.set_initial_value(y0, t0)
        while integrator.successful() and integrator.t < t_max:
            t = integrator.t + step
            p = integrator.integrate(integrator.t + step)
            times.append(t)
            pos.append(p)
        return (times, pos)

    ret = []
    rs = crds.R
    zs = crds.Z
    for r, z in crds:
        # for i in range(len(rs)):
        #     r = rs[i]
        #     z = zs[i]
        times = []
        pos = []
        y0 = [r, z, 0]
        t0 = 0.
        pos.append(y0)
        times.append(t0)

        integrator = ode(b_norm).set_integrator('dopri5')
        integrator.set_initial_value(y0, t0)
        while integrator.successful() and integrator.t < t_max:
            t = integrator.t + step
            p = integrator.integrate(integrator.t + step)
            times.append(t)
            pos.append(p)
        foo = do_step(r, z)
        ret.append(foo)
    # pool = multiprocessing.Pool(3)
    # ret = zip(*pool.map(do_step, [(r, z) for (r, z) in crds]))

    return ret


def dphi_tracer_factory(BR_func, BZ_func, Bphi_func, direction=1):
    """Factory for function $d[R,Z]/d\\phi=f(\\phi, [R,Z])$
    
    The created function is suitable for use in an ODE integrator
    where the toroidal angle $\\phi$ plays the role of time.
    
    Parameters
    ----------
    BR_func, BZ_func, Bphi_func : 
    :param direction: if positive trace field line in/cons the direction of magnetic field.

    Returns
    -------
    dphi_func: function (phi: float, X: [float, float]) -> [float, float]
        gradient function in the [R, Z] plane
       
        
    Note
    ----
    This function is mostly useful when the full spatial coordinates of the field line are required.
    """

    def dphi_func(phi, x):
        R, Z = x
        BR = BR_func(R, Z)
        BZ = BZ_func(R, Z)
        Bphi = Bphi_func(R, Z)
        dRdphi = R * BR / Bphi
        dZdphi = R * BZ / Bphi
        return np.sign(direction)*np.reshape([dRdphi, dZdphi], (2,))  # TODO HOTFIX required when functions return 1d arrays

    return dphi_func


def ds_tracer_factory(BR_func, BZ_func, Bphi_func):
    """Factory for function $d[R,Z]/ds=f(s, [R,Z])$
    
    The created function is suitable for use in an ODE integrator
    where the field line length $s$ plays the role of time.
    
    Parameters
    ----------
    BR_func, BZ_func, Bphi_func : function (R: float, Z:float) -> float
        functions for calculating magnetic field components in the [R, Z] plane
        
    Returns
    -------
    ds_func: function (s: float, X: [float, float]) -> [float, float]
        gradient function in the [R, Z] plane
        
    Note
    ----
    This function is mostly useful when only the field line length is required
    """

    def ds_func(s, x):
        R, Z = x
        BR = BR_func(R, Z)
        BZ = BZ_func(R, Z)
        Bphi = Bphi_func(R, Z)
        B = np.sqrt(np.sum(np.square([BR, BZ, Bphi])))
        dRds = BR / B
        dZds = BZ / B
        return np.reshape([dRds, dZds], (2,))  # TODO HOTFIX required when functions return 1d arrays

    return ds_func


def poloidal_angle_stopper_factory(y0, y_center, direction, stop_res=np.pi / 360):
    """Factory for function which stops field line tracing close to the original poloidal angle
    Suitable for the *events* argument of :func:`scipy.integrate.solve_ivp`
    
    Parameters
    ----------
    y0 : [float, float]
        initial [R, Z] condition for tracing
    y_center : [float, float]
        polar coordinate center [R0, Z0] (e.g. mag. axis)
    direction : float
        sign of dtheta/dphi derivative, depends on plasma current and toroidal mag. field
    stop_res : float
        stopping offset to stop before initial position
        necessary to prevent stopping at initial position
    """
    y_center = np.asarray(y_center)  # should be [R0, Z0]

    def full_arc(y):
        Dy = y - y_center
        theta = np.arctan2(Dy[1], Dy[0])
        theta = np.remainder(theta, 2 * np.pi)
        return theta

    theta_start = full_arc(np.asarray(y0))
    theta_target = theta_start - direction * stop_res
    theta_target = np.remainder(theta_target, 2 * np.pi)

    def stopper(t, y):
        theta = full_arc(y)
        dth = theta - theta_target
        return np.squeeze(dth)

    stopper.terminal = True
    stopper.direction = direction
    return stopper


def z_coordinate_stopper_factory(z_0):
    """
    Factory for function which stops field line tracing close to the defined z planes.
    Suitable for the *events* argument of :func:`scipy.integrate.solve_ivp`

    :param z_0: [float, float]
        [Z_bottom, Z_upper] boundary
    :return:
    """

    def stopper(t, y):
        dist = [y[1] - z_0[0], z_0[1] - y[1]]
        min = np.min(dist)

        return min

    stopper.terminal = True

    return stopper


def rz_coordinate_stopper_factory(r_0, z_0):
    """
    Factory for function which stops field line tracing close to the defined z planes.
    Suitable for the *events* argument of :func:`scipy.integrate.solve_ivp`

    :param r_0: [float, float]
        [R_bottom, R_upper] boundary
    :param z_0: [float, float]
        [Z_bottom, Z_upper] boundary
    :return:
    """

    def stopper(t, y):
        dist = [y[0] - r_0[0], r_0[1] - y[0], y[1] - z_0[0], z_0[1] - y[1]]
        min = np.min(dist)

        return min

    stopper.terminal = True

    return stopper


def ds_grad_psi_tracer_factory(psi_spl):
    """Trace perpendicular to the gradient vector of the psi function

    equivalent to tracing along the poloidal mag. field,
    but the needless 1/R (cancels) and other factors are neglected,
    hopefully making this faster
    """
    def ds_func(s, x):
        R, Z = x
        dpsi_dx = psi_spl(R, Z, dx=1, grid=False)
        dpsi_dy = psi_spl(R, Z, dy=1, grid=False)
        dpsi_ds = (dpsi_dx**2 + dpsi_dy**2)**0.5
        dy_ds = - dpsi_dx / dpsi_ds  # negative to be perp to gradient
        dx_ds = dpsi_dy / dpsi_ds
        return [dx_ds, dy_ds]

    return ds_func


def rz_target_s_min_stopper_factory(y0, s_min, atol=1e-6):
    """Stopper which returns the (squared) distance after a minimum traced length

    atol is subtracted from the distance to guarantee 0-crossing


    This should prevent early stopping by having the initial distance 0

    A good estimate for s_min could be r_mid of y0=[Rt, Zt],
    because the total flux surface length will be at least pi times larger
    """
    def stopper(s, y):
        if s < s_min:           # only started tracing
            return 42
        else:
            return np.sum((y-y0)**2) - atol

    stopper.terminal = True

    return stopper
