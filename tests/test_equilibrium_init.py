"""Tests of the `Equilibrium` initialisation: hints, init_method, synthetic wall,
vacuum equilibria, profile handling, logging and the failure path."""

import logging

import numpy as np
import pytest

import pleque.utils.equi_tools as eq_tools
from pleque import Equilibrium
from pleque.config.settings import reload_settings
from pleque.io.readers import read_geqdsk
from pleque.tests.utils import get_test_equilibria_filenames, synthetic_dataset, synthetic_test_wall

MG_AXIS = (1.5, 0.2)
PSI_LCFS = 0.25


@pytest.fixture
def synthetic_eq():
    return Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall())


def test_basic_construction(synthetic_eq):
    eq = synthetic_eq
    assert np.allclose(eq._mg_axis, MG_AXIS, atol=1e-3)
    assert np.isclose(eq._psi_axis, 0.0, atol=1e-8)
    assert np.isclose(np.asarray(eq._psi_lcfs).item(), PSI_LCFS, atol=1e-6)
    assert eq._limiter_plasma
    assert not eq._vacuum
    assert len(eq._lcfs) > 10


def test_synthetic_wall_generation():
    from pleque.config.settings import get_settings

    ds = synthetic_dataset()
    eq = Equilibrium(ds)

    wall_cfg = get_settings().grid
    fw = eq._first_wall
    assert len(fw) == 4 * wall_cfg.synthetic_wall_points_per_side
    # the synthesized wall is slightly inset with respect to the grid
    assert np.min(fw[:, 0]) > np.min(ds.R.values)
    assert np.max(fw[:, 0]) < np.max(ds.R.values)
    assert np.min(fw[:, 1]) > np.min(ds.Z.values)
    assert np.max(fw[:, 1]) < np.max(ds.Z.values)


def test_synthesize_rectangular_wall_unit():
    rs = np.linspace(1.0, 2.0, 10)
    zs = np.linspace(-1.0, 1.0, 20)
    wall = eq_tools.synthesize_rectangular_wall(rs, zs)
    assert wall.ndim == 2 and wall.shape[1] == 2
    assert np.min(wall[:, 0]) > 1.0 and np.max(wall[:, 0]) < 2.0
    assert np.min(wall[:, 1]) > -1.0 and np.max(wall[:, 1]) < 1.0


def test_vacuum_equilibrium():
    eq = Equilibrium(synthetic_dataset(profiles="none"), first_wall=synthetic_test_wall())
    assert eq._vacuum
    psi_n = np.linspace(0, 1, 5)
    assert np.allclose(eq.pressure(psi_n=psi_n), 0)
    # construction completed: boundary and critical points exist
    assert eq._limiter_plasma
    assert np.allclose(eq._mg_axis, MG_AXIS, atol=1e-3)


def test_profile_input_equivalence():
    """Profiles given as derivatives (pprime, FFprime) vs integrated (p, F) give the same equilibrium."""
    eq_deriv = Equilibrium(synthetic_dataset(profiles="pprime_ffprime"), first_wall=synthetic_test_wall())
    eq_direct = Equilibrium(synthetic_dataset(profiles="p_f"), first_wall=synthetic_test_wall())

    psi_n = np.linspace(0, 1, 20)
    assert np.allclose(eq_deriv.pressure(psi_n=psi_n), eq_direct.pressure(psi_n=psi_n), rtol=1e-3, atol=1e-4)
    assert np.allclose(eq_deriv.F(psi_n=psi_n), eq_direct.F(psi_n=psi_n), rtol=1e-3, atol=1e-4)


def test_hint_kwargs():
    eq = Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall(),
                     mg_axis=MG_AXIS, psi_lcfs=PSI_LCFS,
                     x_points=np.empty((0, 2)), strike_points=None)
    assert np.allclose(eq._mg_axis, MG_AXIS, atol=1e-3)


def test_hints_from_basedata():
    ds = synthetic_dataset()
    ds['mg_axis'] = (('rz',), np.array(MG_AXIS))
    ds['psi_lcfs'] = PSI_LCFS
    eq = Equilibrium(ds, first_wall=synthetic_test_wall())
    assert np.allclose(eq._mg_axis, MG_AXIS, atol=1e-3)


def test_hint_duplicity_warning(caplog):
    ds = synthetic_dataset()
    ds['mg_axis'] = (('rz',), np.array([1.4, 0.1]))
    with caplog.at_level(logging.WARNING, logger="pleque"):
        Equilibrium(ds, first_wall=synthetic_test_wall(), mg_axis=MG_AXIS)
    assert any("mg_axis specified both in basedata and as an argument" in r.message for r in caplog.records)


def test_fast_forward_trusts_hints():
    eq = Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall(),
                     mg_axis=MG_AXIS, psi_lcfs=PSI_LCFS, x_points=np.empty((0, 2)),
                     init_method="fast_forward")
    # hints are taken as final: no critical-point search, exact hint values kept
    assert np.array_equal(eq._mg_axis, np.asarray(MG_AXIS, dtype=float))
    assert eq._x_point is None
    assert len(eq._o_points) == 1
    assert np.asarray(eq._psi_lcfs).item() == PSI_LCFS
    assert eq._limiter_plasma


def test_fast_forward_fallback_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="pleque"):
        eq = Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall(), init_method="fast_forward")
    assert any("falling back to 'hints'" in r.message for r in caplog.records)
    assert np.allclose(eq._mg_axis, MG_AXIS, atol=1e-3)


def test_invalid_init_method():
    with pytest.raises(ValueError, match="init_method"):
        Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall(), init_method="bogus")


@pytest.mark.parametrize("case", [0, 3])
def test_init_method_consistency(case):
    """'full', 'hints' and 'fast_forward' (without hints: fallback) agree on bundled cases."""
    gfile = get_test_equilibria_filenames()[case]
    eq_full = read_geqdsk(gfile, init_method="full")
    eq_hints = read_geqdsk(gfile, init_method="hints")
    eq_fast = read_geqdsk(gfile, init_method="fast_forward")

    for eq in (eq_full, eq_fast):
        assert np.allclose(eq._mg_axis, eq_hints._mg_axis)
        assert np.isclose(eq._psi_axis, eq_hints._psi_axis)
        assert np.isclose(np.asarray(eq._psi_lcfs).item(), np.asarray(eq_hints._psi_lcfs).item())
        if eq_hints._x_point is not None:
            assert np.allclose(eq._x_point, eq_hints._x_point)
        if eq_hints._strike_points is not None and eq._strike_points is not None:
            assert np.allclose(eq._strike_points, eq_hints._strike_points)


def test_double_null_x_points():
    gfile = get_test_equilibria_filenames()[2]
    eq = read_geqdsk(gfile)
    assert eq._x_point is not None
    assert eq._x_point2 is not None
    psi1 = np.asarray(eq._spl_psi(*eq._x_point, grid=False)).item()
    psi2 = np.asarray(eq._spl_psi(*eq._x_point2, grid=False)).item()
    psi_span = abs(np.asarray(eq._psi_lcfs).item() - eq._psi_axis)
    assert abs(psi1 - psi2) / psi_span < 0.05


def test_spline_order_override():
    eq = Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall(), spline_order=5, spline_smooth=0)
    assert eq._spl_psi.degrees == (5, 5)


def test_init_logging(caplog):
    with caplog.at_level(logging.DEBUG, logger="pleque"):
        Equilibrium(synthetic_dataset())
    messages = [r.message for r in caplog.records]
    assert any("Limiter plasma found." in m for m in messages)
    assert any("rectangular wall" in m for m in messages)


def test_verbose_smoke():
    eq = Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall(), verbose=True)
    assert eq._verbose
    # restore the default level changed by verbose=True
    logging.getLogger("pleque").setLevel(logging.WARNING)


def _dataset_without_profiles_coord():
    """Dataset that fails the initialisation after the spatial data are loaded (no psi_n)."""
    ds = synthetic_dataset()
    return ds.drop_vars(["pprime", "FFprime"]).drop_vars("psi_n")


def test_failure_is_logged_and_no_debug_plot(caplog, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    with caplog.at_level(logging.ERROR, logger="pleque"):
        with pytest.raises(AttributeError):
            Equilibrium(_dataset_without_profiles_coord(), first_wall=synthetic_test_wall())

    assert any("Equilibrium initialization failed." in r.message for r in caplog.records)
    assert shown == []


def test_failure_debug_plot_enabled(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    monkeypatch.setenv("PLEQUE_DEBUG_PLOTS", "1")
    reload_settings()
    try:
        with pytest.raises(AttributeError):
            Equilibrium(_dataset_without_profiles_coord(), first_wall=synthetic_test_wall())
        assert shown == [True]
    finally:
        monkeypatch.delenv("PLEQUE_DEBUG_PLOTS")
        reload_settings()
        plt.close("all")
