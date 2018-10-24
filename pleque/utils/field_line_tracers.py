from pleque import Equilibrium
import numpy as np

def _trace_field_line_first_attempt(eq: Equilibrium, *coordinates, coord_type=None, sign=1, step=5e-1, t_max=150, **coords):
    """Deprecated.
    :param coordinates:
    :param coord_type:
    :param coords:
    :return:
    """
    # from numba import autojit
    import multiprocessing
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
