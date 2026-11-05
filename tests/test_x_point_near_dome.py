"""Regression tests of the `Equilibrium` initialisation for X-point plasmas whose X-point is close to the wall.

`pleque.tests.utils.synthetic_x_point_near_dome` builds a line-current model of a lower single null with the lower
X-point a given distance above a divertor dome and a second X-point about 1 m above it, mimicking COMPASS-U MEQ
downward-VDE equilibria on which two failures used to occur:

* the refinement of the lower X-point candidate jumped to the upper X-point, so the lower X-point was lost,
  a wall point outside the separatrix was taken for the limiter point and no closed LCFS was found;
* a dome point in the private flux region (psi_n slightly below 1) passed the coarse psi-monotonicity test,
  so the diverted plasma was classified as dome-limited.
"""

import logging

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

import pleque.utils.equi_tools as eq_tools
from pleque import Equilibrium
from pleque.tests.utils import synthetic_dataset, synthetic_test_wall, synthetic_x_point_near_dome

DIVERTED_DOME_GAPS = [0.02, 0.006, 0.004, 0.002, 0.001]  # X-point above the dome [m]
LIMITED_DOME_GAPS = [-0.005, -0.02]  # dome top above the X-point [m]


def _psi_spline(basedata):
    r, z = basedata.R.values, basedata.Z.values
    return RectBivariateSpline(r, z, basedata.psi.transpose("R", "Z").values, kx=3, ky=3, s=0)


@pytest.mark.parametrize("dome_gap", DIVERTED_DOME_GAPS)
def test_x_point_above_dome_is_diverted(dome_gap):
    basedata, first_wall, truth = synthetic_x_point_near_dome(dome_gap)
    eq = Equilibrium(basedata, first_wall=first_wall)

    assert not eq._limiter_plasma
    np.testing.assert_allclose(eq._x_point, truth["x_points"][0], atol=1e-3)
    np.testing.assert_allclose(eq._x_point2, truth["x_points"][1], atol=1e-3)
    rel_err = (float(eq._psi_lcfs) - truth["psi_x_point"]) / (truth["psi_x_point"] - truth["psi_axis"])
    assert abs(rel_err) < 1e-3
    assert eq._lcfs is not None and len(eq._lcfs) > 10


@pytest.mark.parametrize("dome_gap", LIMITED_DOME_GAPS)
def test_dome_above_x_point_is_limited(dome_gap):
    basedata, first_wall, truth = synthetic_x_point_near_dome(dome_gap)
    eq = Equilibrium(basedata, first_wall=first_wall)

    assert eq._limiter_plasma
    # the plasma touches the dome top
    dome_top = truth["x_points"][0, 1] - dome_gap
    assert abs(eq._limiter_point[1] - dome_top) < 1e-3
    assert 0.70 <= eq._limiter_point[0] <= 0.74
    np.testing.assert_allclose(eq._x_point2, truth["x_points"][1], atol=1e-3)


def test_critical_point_refinement_stays_local():
    """Refining a grid X-point candidate must not jump to the other X-point 1 m away."""
    basedata, _, truth = synthetic_x_point_near_dome(0.01)
    r, z = basedata.R.values, basedata.Z.values
    psi_spl = _psi_spline(basedata)

    x_candidates, _ = eq_tools.find_extremes(r, z, psi_spl)
    assert len(x_candidates) == 2
    grad_sq = eq_tools._make_psi_grad_sq(psi_spl)
    for candidate in x_candidates:
        refined = eq_tools.minimize_in_vicinity(candidate, grad_sq, (r[0], r[-1]), (z[0], z[-1]))
        nearest = truth["x_points"][np.argmin(np.hypot(*(truth["x_points"] - candidate).T))]
        np.testing.assert_allclose(refined, nearest, atol=1e-3)


@pytest.mark.parametrize("dome_gap", [0.004, 0.002])
def test_dome_point_below_x_point_is_not_limiter(dome_gap):
    """With the X-point known, a dome point a few mm below it (private flux region) must not limit the plasma."""
    basedata, first_wall, truth = synthetic_x_point_near_dome(dome_gap)
    x_point = truth["x_points"][0]

    limited, point = eq_tools.recognize_plasma_type(x_point, first_wall, truth["mg_axis"], truth["psi_axis"],
                                                    _psi_spline(basedata))
    assert not limited
    np.testing.assert_allclose(point, x_point)


def _saddle(r, z, grid=False):
    """psi with an X-point at the origin; psi decreases from it both up (core) and down (private flux region)."""
    return np.asarray(r) ** 2 - np.asarray(z) ** 2


def test_is_monotonic_refinement_detects_extremum_near_start():
    wall_point = (0.0, -0.004)  # private flux region, 4 mm below the X-point
    axis = (0.0, 0.5)
    # equidistant sampling (~1 cm) steps over the X-point
    assert eq_tools.is_monotonic(_saddle, wall_point, axis, 50)
    assert not eq_tools.is_monotonic(_saddle, wall_point, axis, 50, refine_start=1e-4)
    # truly monotonic profiles stay monotonic with the refinement
    assert eq_tools.is_monotonic(_saddle, (0.0, 0.004), axis, 50, refine_start=1e-4)
    assert eq_tools.is_monotonic(_saddle, (0.0, 0.004), axis, 50, refine_start=1.0)  # longer than the segment


def test_recognize_x_points_discards_duplicate(monkeypatch, caplog):
    psi_spl = RectBivariateSpline(np.linspace(0, 1, 10), np.linspace(-1, 1, 10), np.zeros((10, 10)))
    monkeypatch.setattr(eq_tools, "minimize_in_vicinity", lambda point, *args, **kwargs: np.array([0.5, -0.5]))
    x_points = np.array([[0.5, -0.5], [0.5, 0.5]])
    with caplog.at_level(logging.WARNING, logger="pleque"):
        (xp1, xp2), _ = eq_tools.recognize_x_points(x_points, np.array([0.5, 0.0]), 0.0, psi_spl, (0, 1), (-1, 1))
    np.testing.assert_allclose(xp1, [0.5, -0.5])
    assert xp2 is None
    assert "converged to the same point" in caplog.text


def test_missing_closed_lcfs_raises_clear_error(monkeypatch):
    monkeypatch.setattr(eq_tools, "find_close_lcfs", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="No closed flux surface enclosing the magnetic axis"):
        Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall())
