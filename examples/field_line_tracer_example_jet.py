import matplotlib.pyplot as plt
import numpy as np

from pleque.utils.field_line_tracers import _trace_field_line_first_attempt
from pleque.utils.plotting import plot_equilibrium
from pleque_test.testing_utils import load_testing_equilibrium
from pleque.io import jet

def first_attempt():
    eq = jet.sal_jet(92400)

    plot_equilibrium(eq)
    # plt.show()

    # N = 20
    N = 5

    rs = np.linspace(1.16, 1.17, N)
    zs = np.zeros_like(rs)
    traces = _trace_field_line_first_attempt(eq, R=rs, Z=zs, step=1e-2)

    # traces = eq._trace_field_line(R=1.17, Z=0)

    def convert_to_cart(r, z, phi):
        x = r * np.sin(phi)
        y = r * np.cos(phi)
        z = z
        return x, y, z

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for trace in traces:
        p = trace[0]
        trace_array = np.array(trace[1]).squeeze()
        r = trace_array[:, 0]
        z = trace_array[:, 1]
        phi = trace_array[:, 2]
        x, y, z = convert_to_cart(r, z, phi)
        # ax.plot(x, y, z)
        ax.scatter(x, y, z, c=p, s=0.3, marker='.')

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    # plt.savefig('/compass/home/kripner/Desktop/tracinkg/trace_1.png')

    fig = plt.figure()
    ax = fig.gca()

    sc = None
    for trace in traces:
        p = trace[0]
        trace_array = np.array(trace[1]).squeeze()
        r = trace_array[:, 0]
        z = trace_array[:, 1]
        phi = trace_array[:, 2]
        x, y, z = convert_to_cart(r, z, phi)
        # ax.plot(x, y, z)
        sc = ax.scatter(r, z, c=p, s=0.3, marker='.')

    if sc is not None:
        plt.colorbar(sc)
    ax.set_aspect('equal')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    print('End of tracing')


def default_tracer():
    eq = load_testing_equilibrium()

    N = 1
    rs = np.linspace(1.16, 1.17, N, endpoint=False)
    zs = np.zeros_like(rs)

    traces = eq.trace_field_line(R=rs, Z=zs)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for fl in traces:
        ax.scatter(fl.X, fl.Y, fl.Z, s=0.3, marker='.')

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    fig = plt.figure()
    ax = fig.gca()

    for fl in traces:
        ax.scatter(fl.R, fl.Z, s=0.3, marker='.')

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')

    fig = plt.figure()
    ax = fig.gca()

    for fl in traces:
        ax.plot(fl.X, fl.Y)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')


if __name__ == '__main__':
    # first_attempt()
    default_tracer()

    plt.show()
