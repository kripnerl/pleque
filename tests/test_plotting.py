"""Tests of the equilibrium plotting routines (`pleque.utils.plotting`)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pleque.tests.utils import load_testing_equilibrium
from pleque.utils.plotting import _layer_kwargs, plot_extremes, plot_selected_contours

#: `baseline_eqdsk`: diverted, 6 X-points, secondary X-point at psi_n ~ 1.036, no first wall.
DIVERTED_CASE = 0
#: `g13127.1050`: limiter plasma with a full first wall and no X-point at all.
WALLED_CASE = 3


@pytest.fixture(scope="module")
def diverted_eq():
    return load_testing_equilibrium(DIVERTED_CASE)


@pytest.fixture(scope="module")
def walled_eq():
    return load_testing_equilibrium(WALLED_CASE)


@pytest.fixture
def ax():
    fig, axis = plt.subplots()
    yield axis
    plt.close(fig)


def _lines_with_color(axis, color):
    return [line for line in axis.lines if line.get_color() == color]


# --------------------------------------------------------------------------------------
# Layer specification resolver
# --------------------------------------------------------------------------------------


def test_layer_kwargs_none_and_true_give_defaults():
    defaults = {"color": "C1", "lw": 2}

    assert _layer_kwargs(None, defaults) == defaults
    assert _layer_kwargs(True, defaults) == defaults


def test_layer_kwargs_false_disables_the_layer():
    assert _layer_kwargs(False, {"color": "C1"}) is None


def test_layer_kwargs_mapping_is_merged_over_defaults():
    assert _layer_kwargs({"color": "k"}, {"color": "C1", "lw": 2}) == {"color": "k", "lw": 2}


def test_layer_kwargs_does_not_mutate_the_defaults():
    defaults = {"color": "C1"}
    _layer_kwargs({"color": "k"}, defaults)

    assert defaults == {"color": "C1"}


def test_layer_kwargs_rejects_other_types():
    with pytest.raises(TypeError):
        _layer_kwargs("solid", {"color": "C1"})


# --------------------------------------------------------------------------------------
# Overview plot
# --------------------------------------------------------------------------------------


def test_plot_overview_runs_for_all_test_equilibria(equilibrium, ax):
    """Smoke test over all the bundled equilibria."""
    assert equilibrium.plot_overview(ax=ax) is ax
    assert len(ax.lines) > 0
    assert len(ax.collections) > 0
    assert ax.get_xlabel() == "R [m]"


def test_plot_overview_draws_the_default_layers(walled_eq, ax):
    """Regression guard: the default figure must keep all the layers it had."""
    walled_eq.plot_overview(ax=ax)

    labels = [line.get_label() for line in ax.lines]
    assert "First wall" in labels
    assert _lines_with_color(ax, "C3"), "separatrix is missing"
    assert _lines_with_color(ax, "C1"), "LCFS is missing"
    assert _lines_with_color(ax, "royalblue"), "magnetic axis marker is missing"
    # psi contours, near-SOL contours and the shaded first wall interior
    assert len(ax.collections) == 3


def test_disabled_layers_are_not_drawn(walled_eq, ax):
    walled_eq.plot_overview(
        ax=ax, first_wall=False, separatrix=False, lcfs=False, psi_contours=False, near_sol=False, extremes=False
    )

    assert len(ax.lines) == 0
    assert len(ax.collections) == 0


def test_single_layer_can_be_disabled(walled_eq, ax):
    walled_eq.plot_overview(ax=ax, near_sol=False)

    # one contour set less than the default figure
    assert len(ax.collections) == 2


def test_layer_style_is_applied(walled_eq, ax):
    walled_eq.plot_overview(ax=ax, lcfs={"color": "k", "lw": 1})

    assert not _lines_with_color(ax, "C1"), "the LCFS default colour was not overridden"
    restyled = [line for line in _lines_with_color(ax, "k") if line.get_linewidth() == 1]
    assert len(restyled) == 1


def test_psi_contour_levels_can_be_selected(walled_eq, ax):
    """`levels` used to collide with the positional argument of `ax.contour`."""
    levels = sorted(float(walled_eq.psi(psi_n=psi_n)[0]) for psi_n in (0.3, 0.6, 0.9))

    walled_eq.plot_overview(ax=ax, psi_contours={"levels": levels}, near_sol=False)

    psi_contours = ax.collections[-1]
    assert np.allclose(psi_contours.levels, levels)


def test_dr_sol_still_works(walled_eq, ax):
    """The deprecated `dr_sol` argument keeps working."""
    walled_eq.plot_overview(ax=ax, dr_sol=5e-3)

    assert len(ax.collections) == 3


# --------------------------------------------------------------------------------------
# Selected contours
# --------------------------------------------------------------------------------------


def test_selected_contour_from_a_number(diverted_eq, ax):
    cs = plot_selected_contours(diverted_eq, ax, 1.0)

    assert np.allclose(cs.levels, [1.0])


def test_selected_contour_from_a_named_point(diverted_eq, ax):
    cs = plot_selected_contours(diverted_eq, ax, "secondary_x_point")

    assert np.allclose(cs.levels, diverted_eq.secondary_x_point.psi_n)


def test_selected_contours_from_a_mixed_sequence(diverted_eq, ax):
    cs = plot_selected_contours(diverted_eq, ax, [1.0, "secondary_x_point"])

    expected = sorted({1.0, float(diverted_eq.secondary_x_point.psi_n[0])})
    assert np.allclose(cs.levels, expected)


def test_selected_contour_from_coordinates(diverted_eq, ax):
    point = diverted_eq.coordinates(psi_n=0.5)
    cs = plot_selected_contours(diverted_eq, ax, point)

    assert np.allclose(cs.levels, [0.5])


def test_selected_contours_accept_a_mapping(diverted_eq, ax):
    cs = plot_selected_contours(diverted_eq, ax, {"levels": 1.0, "colors": "magenta"})

    assert np.allclose(cs.levels, [1.0])
    assert cs.get_edgecolor().shape[0] == 1


def test_missing_named_point_is_skipped_with_a_warning(walled_eq, ax, caplog):
    with caplog.at_level("WARNING", logger="pleque.utils.plotting"):
        cs = plot_selected_contours(walled_eq, ax, "secondary_x_point")

    assert cs is None
    assert len(ax.collections) == 0
    assert "secondary_x_point" in caplog.text


def test_unknown_contour_name_raises(diverted_eq, ax):
    with pytest.raises(ValueError, match="Unknown contour name"):
        plot_selected_contours(diverted_eq, ax, "tertiary_x_point")


def test_unsupported_contour_type_raises(diverted_eq, ax):
    with pytest.raises(TypeError):
        plot_selected_contours(diverted_eq, ax, [object()])


def test_true_is_not_a_contour_specification(diverted_eq, ax):
    """`bool` is a `Number`, so `True` would silently become the psi_n = 1 contour."""
    with pytest.raises(TypeError):
        plot_selected_contours(diverted_eq, ax, True)


def test_false_selects_no_contours(diverted_eq, ax):
    assert plot_selected_contours(diverted_eq, ax, False) is None
    assert len(ax.collections) == 0


def test_no_contours_selected_draws_nothing(diverted_eq, ax):
    assert plot_selected_contours(diverted_eq, ax, None) is None
    assert len(ax.collections) == 0


def test_selected_contours_are_drawn_below_all_other_layers(diverted_eq, ax):
    diverted_eq.plot_overview(ax=ax, contours=1.0)

    children = ax.get_children()
    selected = ax.collections[0]

    assert np.allclose(selected.levels, [1.0])
    assert children.index(selected) == 0
    assert all(children.index(selected) < children.index(line) for line in ax.lines)


# --------------------------------------------------------------------------------------
# Extremes
# --------------------------------------------------------------------------------------


def test_plot_extremes_honours_all_x_points(diverted_eq, ax):
    """`all_x_points` used to be overridden by `all_o_points`."""
    assert len(diverted_eq._x_points) > 2, "the test case has to have more than two X-points"

    plot_extremes(diverted_eq, ax, all_o_points=True, all_x_points=False)
    limited = _lines_with_color(ax, "crimson")[0]
    assert len(limited.get_xdata()) == 2

    ax.clear()
    plot_extremes(diverted_eq, ax, all_o_points=True, all_x_points=True)
    unlimited = _lines_with_color(ax, "crimson")[0]
    assert len(unlimited.get_xdata()) == len(diverted_eq._x_points)


# --------------------------------------------------------------------------------------
# Geometry (COCOS) plot
# --------------------------------------------------------------------------------------


def test_plot_geometry_returns_both_views(diverted_eq):
    axs = diverted_eq.plot_geometry()

    assert len(axs) == 2
    assert axs[0].get_title() == "Top view"
    assert axs[1].get_title() == "Poloidal cross section"
    plt.close("all")


@pytest.mark.parametrize("view, title", [("top_view", "Poloidal cross section"), ("poloidal_view", "Top view")])
def test_plot_geometry_single_view(diverted_eq, view, title):
    axs = diverted_eq.plot_geometry(**{view: False})

    assert len(axs) == 1
    assert axs[0].get_title() == title
    plt.close("all")


def test_plot_geometry_without_any_view_raises(diverted_eq):
    with pytest.raises(ValueError):
        diverted_eq.plot_geometry(top_view=False, poloidal_view=False)


def test_plot_geometry_text_kwargs_are_forwarded(diverted_eq):
    """`**kwargs` used to be documented but silently dropped."""
    axs = diverted_eq.plot_geometry(text_kwargs={"fontsize": 20})

    assert axs[0].texts
    assert all(text.get_fontsize() == 20 for text in axs[0].texts)
    plt.close("all")


def test_plot_geometry_layers_can_be_disabled(diverted_eq):
    axs = diverted_eq.plot_geometry(top_view=False, magnetic_axis=False, first_wall=False)

    assert len(axs[0].lines) == 0
    plt.close("all")
