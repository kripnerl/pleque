from pleque import Coordinates
import numpy as np


def test_normal_vectors():

    n_points = 30

    # Vertical line and its normal
    line = Coordinates(None, R=np.linspace(0, 1, n_points), Z=np.zeros(n_points))
    line_normals = line.normal_vector()

    assert line_normals.shape == (3, n_points)
    assert np.allclose(line_normals[0, :], 0)
    assert np.allclose(line_normals[1, :], 1)
    assert np.allclose(line_normals[2, :], 0)

    # Diagonal line and its normal
    line = Coordinates(None, R=np.linspace(0, 1, n_points), Z=np.linspace(0, 1, n_points))
    line_normals = line.normal_vector()

    assert np.allclose(line_normals[2, :], 0)
    assert np.allclose(line_normals[0, :], -line_normals[1, :])
    assert np.allclose(np.linalg.norm(line_normals, axis=0), 1)


def test_incidence_angle_sin():

    n_points = 30

    # Vertical line:
    line = Coordinates(None, R=np.linspace(0, 1, n_points), Z=np.zeros(n_points))

    # Generate vectors in various angles (from [0, 2, 0] to [2, 0, 0])
    rvec = np.linspace(0, 2, n_points, endpoint=True)
    zvec = np.linspace(2, 0, n_points, endpoint=True)

    vecs = np.array([rvec, zvec, np.zeros_like(rvec)])

    assert vecs.shape == (3, n_points)

    inc_angls = line.incidence_angle_cos(vecs)

    assert np.isclose(inc_angls[0], 1)
    assert np.isclose(inc_angls[-1], 0)
    assert np.all(np.diff(inc_angls) < 0)


def test_imact_angles(equilibrium):

    fw = equilibrium.first_wall

    if len(fw) and len(fw) > 4:

        sines = fw.impact_angle_sin()
        pol_sin = fw.impact_angle_sin_pol_projection()

        assert np.all(np.abs(sines) < 0.3)
        assert np.all(np.abs(pol_sin) < 1)
