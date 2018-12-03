import matplotlib.pyplot as plt
import numpy as np

from pleque_test.testing_utils import load_testing_equilibrium

def default_tracer():
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    eq = load_testing_equilibrium()

    N = 1
    rs = np.linspace(1.16, 1.17, N, endpoint=False)
    zs = np.zeros_like(rs)

    traces = eq.trace_field_line(R=rs, Z=zs)

    dists, lines = eq.connection_length(R=1.18, Z=0)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for fl in traces:
        ax.scatter(fl.X, fl.Y, fl.Z, s=0.3, marker='.')
    for fl in lines:
        ax.scatter(fl.X, fl.Y, fl.Z, s=0.3, marker='.')

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    fig = plt.figure()
    ax = fig.gca()

    for fl in traces:
        ax.scatter(fl.R, fl.Z, s=0.3, marker='.')
    for fl in lines:
        ax.scatter(fl.R, fl.Z, s=0.3, marker='.')

    print(dists)

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
