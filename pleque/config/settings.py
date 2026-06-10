"""
Central configuration for PLEQUE.

All tunable algorithm parameters (grid resolutions, solver tolerances, search
heuristics, plotting defaults, ...) live in the :class:`Settings` model below,
grouped into per-topic sections. Values are resolved with the following
precedence (highest first):

1. keyword arguments passed directly to ``Settings(...)``,
2. environment variables (``PLEQUE_`` prefix, ``__`` as section delimiter,
   e.g. ``PLEQUE_FLUX_SURFACES__N_PSI=300``),
3. a single configuration file -- the first one found of:

   a. the file pointed to by the ``PLEQUE_CONFIG_FILE`` environment variable,
   b. ``pleque.toml`` in the current working directory,
   c. ``pleque.toml`` in the user configuration directory
      (``%APPDATA%\\pleque\\`` on Windows, ``$XDG_CONFIG_HOME/pleque/`` or
      ``~/.config/pleque/`` elsewhere),
   d. a ``[tool.pleque]`` table in the nearest ``pyproject.toml`` found by
      walking up from the current working directory,

4. the hard-coded defaults defined in this module.

Configuration files are *not* merged: the first existing file wins and the
remaining locations are ignored. The winning file (if any) is recorded on the
settings object as :attr:`Settings.config_source`.

See ``docs/source/configuration.rst`` for the full reference.
"""

import math
import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_FILE_NAME = "pleque.toml"
CONFIG_FILE_ENV_VAR = "PLEQUE_CONFIG_FILE"


class GridSettings(BaseModel):
    """Default computational (R, Z) grids."""

    default_nr: int = Field(1000, description="Number of R points of the default `Equilibrium.grid()` grid.")
    default_nz: int = Field(2000, description="Number of Z points of the default `Equilibrium.grid()` grid.")
    synthetic_wall_margin: float = Field(
        0.01,
        description="Fractional inset of the rectangular first wall synthesized when no wall is given, "
        "relative to the computational grid size.",
    )
    synthetic_wall_points_per_side: int = Field(
        20, description="Number of points per side of the synthesized rectangular first wall."
    )


class SplineSettings(BaseModel):
    """Spline construction parameters."""

    psi_order: int = Field(3, description="Order of the 2D psi(R, Z) `RectBivariateSpline`.")
    psi_smooth: float = Field(0.0, description="Smoothing factor `s` of the 2D psi(R, Z) spline.")
    profile_order: int = Field(
        3, description="Order `k` of 1D profile splines (F, pressure, q, ... as functions of psi_n)."
    )
    profile_smooth: float = Field(0.0, description="Smoothing factor `s` of 1D profile splines.")


class CriticalPointSettings(BaseModel):
    """Search for critical points of psi (magnetic axis, X-points)."""

    find_extremes_order: int = Field(
        20, description="Number of neighbouring grid points used by `argrelmin` when locating extremes of |grad psi|."
    )
    find_extremes_max_iter: int = Field(
        10, description="Maximum number of retries (with reduced order) when no O-point candidate is found."
    )
    gradient_threshold: float = Field(
        1.0, description="Upper bound on |grad psi|^2 for a grid point to count as a critical-point candidate."
    )
    vicinity_radius: float = Field(
        0.1, description="Half-size [m] of the (R, Z) box used when refining a critical point by local minimization."
    )
    minimizer_xtol: float = Field(1e-7, description="`xtol` passed to the Powell/TNC minimizers refining extremes.")
    relocation_threshold: float = Field(
        1e-2,
        description="If unbounded minimization moves a candidate point by more than this (in psi-grid units), "
        "the bounded minimizer is used instead.",
    )
    axis_vertical_weight: float = Field(
        5.0,
        description="Weight of the radial distance (relative to the vertical one) when ranking O-point candidates "
        "by distance from the expected magnetic-axis position.",
    )
    axis_out_of_wall_penalty: float = Field(
        1e-3, description="Score multiplier penalizing O-point candidates located outside the first wall."
    )
    x_point_sort_epsilon: float = Field(
        1e-3, description="Regularization added to X-point ranking terms to avoid zero scores."
    )
    monotonicity_test_points: int = Field(
        10, description="Number of test points for the psi-monotonicity check between magnetic axis and X-point."
    )
    limiter_monotonicity_test_points: int = Field(
        50, description="Number of test points for the psi-monotonicity check between magnetic axis and limiter point."
    )


class LcfsSettings(BaseModel):
    """Last closed flux surface search and refinement."""

    search_grid_nr: int = Field(700, description="Number of R points of the grid used to locate the LCFS contour.")
    search_grid_nz: int = Field(1200, description="Number of Z points of the grid used to locate the LCFS contour.")
    refinement_tolerance: float = Field(
        1e-10, description="Relative psi error under which the iterative LCFS refinement stops."
    )
    inner_psi_n_offset: float = Field(
        1e-5, description="Offset used as psi_n = 1 - offset when contouring the LCFS as an inner flux surface."
    )
    initial_psi_offset: float = Field(
        1e-4,
        description="Fraction of (psi_lcfs - psi_axis) used to shift psi inwards when searching "
        "for the initial closed LCFS candidate.",
    )
    surface_step_damping: float = Field(
        0.99, description="Damping factor of the gradient step in the iterative flux-surface refinement."
    )
    x_point_shift: float = Field(
        1e-6, description="Distance [m] by which the separatrix tracing start point is shifted from the X-point."
    )


class FieldLineTracingSettings(BaseModel):
    """Field-line tracing (`scipy.integrate.solve_ivp`) parameters."""

    atol: float = Field(1e-6, description="Absolute tolerance of the field-line ODE solver.")
    rtol: float = Field(1e-8, description="Relative tolerance of the field-line ODE solver.")
    x_point_atol_scale: float = Field(
        1e-3,
        description="For X-point plasmas the absolute tolerance is reduced to "
        "min(atol, distance_to_x_point * x_point_atol_scale).",
    )
    max_step: float = Field(1e-2, description="Maximum integration step in toroidal angle phi [rad].")
    max_toroidal_turns: int = Field(50, description="Maximum number of toroidal turns integrated when tracing.")
    poloidal_stop_resolution: float = Field(
        math.pi / 1024,
        description="Poloidal-angle resolution [rad] of the stopper used by `Equilibrium.trace_field_line`.",
    )
    default_stop_resolution: float = Field(
        math.pi / 360, description="Default poloidal-angle resolution [rad] of `poloidal_angle_stopper_factory`."
    )
    target_stopper_atol: float = Field(
        1e-6, description="Distance tolerance [m] of the target-point stopper used when tracing flux surfaces."
    )


class FluxSurfaceSettings(BaseModel):
    """Flux-surface generation, tracing and derived 1D profiles."""

    n_psi: int = Field(200, description="Default number of psi_n levels used to evaluate flux surfaces.")
    psi_n_min: float = Field(0.01, description="Minimum psi_n used for flux-surface averaging and volume integration.")
    contour_step: float = Field(
        1e-3, description="Spatial resolution [m] of flux-surface contours found on the (R, Z) grid."
    )
    contour_grid_nr: int = Field(
        100, description="Default number of R points of the grid used for flux-surface contouring."
    )
    contour_grid_nz: int = Field(
        100, description="Default number of Z points of the grid used for flux-surface contouring."
    )
    trace_step: float = Field(
        1e-3, description="Maximum step [m] along the flux surface used by `Equilibrium.trace_flux_surface`."
    )
    trace_max_turns: int = Field(
        4, description="Maximum number of toroidal turns integrated when tracing a flux surface."
    )
    q_psi_n_min: float = Field(0.01, description="Lowest psi_n of the grid on which the q profile is evaluated.")
    q_psi_n_step: float = Field(0.005, description="psi_n step of the grid on which the q profile is evaluated.")
    q_search_psi_n_max: float = Field(
        0.95, description="Default upper psi_n bound when searching a flux surface with a given q."
    )
    q_search_psi_n_cap: float = Field(
        0.99, description="Hard upper psi_n bound of the root finder used when searching a flux surface with a given q."
    )
    midplane_map_points: int = Field(100, description="Number of points of the midplane r_mid -> psi mapping spline.")


class PlottingSettings(BaseModel):
    """Default parameters of plotting helpers."""

    grid_nr: int = Field(400, description="Number of R points of grids used by plotting routines.")
    grid_nz: int = Field(600, description="Number of Z points of grids used by plotting routines.")
    debug_contour_levels: int = Field(60, description="Number of psi contour levels in the debug overview plot.")
    psi_contour_levels: int = Field(20, description="Default number of contour levels of `plot_psi_contours`.")
    rational_q_psi_n_max: float = Field(
        0.95, description="Upper psi_n bound when looking up rational q surfaces for plotting."
    )
    sol_spacing: float = Field(2e-3, description="Default radial spacing [m] of near-SOL contours.")
    axis_margin: float = Field(
        1 / 12, description="Margin around the first wall, as a fraction of its size, used for axis limits."
    )


class IOSettings(BaseModel):
    """Default parameters of file writers."""

    geqdsk_nx: int = Field(64, description="Default number of R points of a written G-EQDSK file.")
    geqdsk_ny: int = Field(128, description="Default number of Z points of a written G-EQDSK file.")
    geqdsk_nbdry: int = Field(200, description="Default number of boundary points of a written G-EQDSK file.")
    omas_n_psi: int = Field(200, description="Default number of 1D profile points written to an OMAS data structure.")
    omas_grid_step: float = Field(
        1e-3, description="Default (R, Z) grid step [m] of 2D profiles written to an OMAS data structure."
    )


def _windows_config_dir() -> Path:
    """Per-user config dir on Windows: ``%APPDATA%\\pleque`` (``~\\pleque`` if ``APPDATA`` is unset)."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home()
    return base / "pleque"


def _posix_config_dir() -> Path:
    """Per-user config dir elsewhere: ``$XDG_CONFIG_HOME/pleque``, defaulting to ``~/.config/pleque``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "pleque"


def user_config_path() -> Path:
    """Return the per-user config file path (not necessarily existing)."""
    config_dir = _windows_config_dir() if os.name == "nt" else _posix_config_dir()
    return config_dir / CONFIG_FILE_NAME


def _load_pyproject_table(path: Path) -> dict:
    """Return the [tool.pleque] table of a pyproject.toml file ({} if absent or unreadable)."""
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data.get("tool", {}).get("pleque", {})


def _find_project_pyproject() -> Path | None:
    """Walk up from cwd to the first pyproject.toml containing a [tool.pleque] table."""
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate if _load_pyproject_table(candidate) else None
    return None


class _TomlTableSettingsSource(PydanticBaseSettingsSource):
    """Settings source backed by an in-memory mapping (a TOML table read beforehand)."""

    def __init__(self, settings_cls: type[BaseSettings], data: dict):
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(self, field, field_name):
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict:
        return dict(self._data)


def resolve_config_file() -> tuple[Path | None, tuple[Path, ...]]:
    """Resolve the configuration file to load.

    Returns ``(winner, consulted)`` where ``winner`` is the first existing
    config file in precedence order (or ``None`` if none exists) and
    ``consulted`` lists all locations that were checked.

    Raises :class:`FileNotFoundError` if ``PLEQUE_CONFIG_FILE`` is set but
    does not point to an existing file.
    """
    explicit = os.environ.get(CONFIG_FILE_ENV_VAR)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{CONFIG_FILE_ENV_VAR} points to a non-existent file: {path}")
        return path, (path,)

    consulted = [Path.cwd() / CONFIG_FILE_NAME, user_config_path()]
    winner = next((p for p in consulted if p.is_file()), None)
    if winner is None:
        pyproject = _find_project_pyproject()
        if pyproject is not None:
            consulted.append(pyproject)
            winner = pyproject
    return winner, tuple(consulted)


class Settings(BaseSettings):
    """
    PLEQUE configuration, grouped into per-topic sections.

    All defaults can be overridden by a ``pleque.toml`` file, a
    ``[tool.pleque]`` table in ``pyproject.toml``, environment variables
    (``PLEQUE_`` prefix, ``__`` section delimiter) or constructor keyword
    arguments — see the module docstring for the precedence rules.
    """

    model_config = SettingsConfigDict(
        env_prefix="PLEQUE_",
        env_nested_delimiter="__",
    )

    default_cocos: int = Field(3, description="COCOS convention assumed when the input does not specify one.")
    grid: GridSettings = Field(default_factory=GridSettings)
    splines: SplineSettings = Field(default_factory=SplineSettings)
    critical_points: CriticalPointSettings = Field(default_factory=CriticalPointSettings)
    lcfs: LcfsSettings = Field(default_factory=LcfsSettings)
    field_line_tracing: FieldLineTracingSettings = Field(default_factory=FieldLineTracingSettings)
    flux_surfaces: FluxSurfaceSettings = Field(default_factory=FluxSurfaceSettings)
    plotting: PlottingSettings = Field(default_factory=PlottingSettings)
    io: IOSettings = Field(default_factory=IOSettings)

    _config_source: Path | None = PrivateAttr(default=None)
    _config_files_consulted: tuple[Path, ...] = PrivateAttr(default=())

    @property
    def config_source(self) -> Path | None:
        """The configuration file values were loaded from, or None for built-in defaults."""
        return self._config_source

    @property
    def config_files_consulted(self) -> tuple[Path, ...]:
        """All configuration file locations checked during loading, in precedence order."""
        return self._config_files_consulted

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        winner, _ = resolve_config_file()
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if winner is not None:
            if winner.name == "pyproject.toml":
                sources.append(_TomlTableSettingsSource(settings_cls, _load_pyproject_table(winner)))
            else:
                sources.append(TomlConfigSettingsSource(settings_cls, winner))
        sources.append(file_secret_settings)
        return tuple(sources)


def load_settings() -> Settings:
    """Build a fresh :class:`Settings` instance (bypassing the cache), recording its source."""
    # The file is resolved a second time inside settings_customise_sources; both
    # resolutions are a few stat() calls and see the same file except for a
    # negligible race window.
    winner, consulted = resolve_config_file()
    settings = Settings()
    settings._config_source = winner
    settings._config_files_consulted = consulted
    return settings


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return load_settings()


def reload_settings() -> Settings:
    """Clear the settings cache and reload from configuration files and environment."""
    get_settings.cache_clear()
    return get_settings()
