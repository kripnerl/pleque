# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PLEQUE (**PL**asma **EQU**ilibrium **E**njoyment module) is a Python library for visualisation and manipulation of tokamak plasma equilibria. The central abstraction is the `Equilibrium` class, built from poloidal magnetic flux `psi(R, Z)` on a rectangular grid plus 1D profiles, typically loaded from G-EQDSK files.

## Commands

The project uses Poetry (build backend: `poetry-core`; Python `^3.10`).

```bash
# Install (the compass group needs a private IPP repo and is skipped in CI;
# omas is also temporarily disabled due to complicated dependencies)
poetry install --without compass

# Run all tests (this is exactly what CI runs — see .gitlab-ci.yml)
pytest

# Run a single test file / test
pytest tests/test_equilibria.py
pytest tests/test_equilibria.py::test_equilibria

# Build docs (Sphinx, published on readthedocs)
make -C docs html

# Build a distribution
poetry build
```

There is no pytest/lint configuration file; `pylint` and `flake8` are dev dependencies but are not enforced by CI.

## Tests

- Tests live in `tests/` and run against six bundled equilibria (G-EQDSK files and one NetCDF/JOREK file) in `pleque/resources/`, accessed via `pleque/tests/utils.py`.
- `tests/conftest.py` defines the module-scoped `equilibrium` fixture, parametrized over test cases 0–5, and a `geqdsk_file` fixture. Most tests just take `equilibrium` as an argument and are automatically run against all bundled equilibria.
- COMPASS database tests (`tests/test_cdb.py`) use `pytest.importorskip('pyCDB')` and are skipped without the private dependency.
- `tests/test_dimenstions.py` auto-discovers and checks the output shapes of all `Equilibrium` methods registered with the `@scalar_function` / `@vector_function` decorators — new public evaluation methods should be registered with those decorators (see below).

## Architecture

### Core classes (`pleque/core/`)

- **`Equilibrium`** (`equilibrium.py`, ~2000 lines) — the heart of the package. Constructed from an `xarray.Dataset` (`basedata`) with `psi(R, Z)`, profiles `pressure`, `F`/`FFprime` over `psi_n`, and optionally a first wall. On init it builds 2D/1D splines (`RectBivariateSpline`/`UnivariateSpline`) and locates critical points (magnetic axis, x-points, strike points). `init_method="hints"` is the only well-tested initialization mode. Exposes evaluation methods (`psi`, `B_R`, `B_tor`, `q`, `j_tor`, ...), geometry properties (`lcfs`, `separatrix`, `first_wall`, `magnetic_axis`, `x_point`, ...), field-line tracing (`trace_field_line`), flux-expansion metrics, plotting (`plot_overview`, `plot_geometry`), and `to_geqdsk` output.
- **`Coordinates`** (`coordinates.py`) — unified coordinate handling. Every evaluation method on `Equilibrium` shares the signature pattern `(self, *coordinates, R=None, Z=None, coord_type=None, grid=..., **coords)` and converts its input through `Equilibrium.coordinates(...)`. Supported systems: 1D `psi_n`/`psi`/`rho`; 2D `(R, Z)` (default) and `(r, theta)`; 3D `(R, Z, phi)` and `(X, Y, Z)`.
- **`Surface` / `FluxSurface`** (`fluxsurface.py`) — subclasses of `Coordinates` wrapping shapely polygons/linestrings; provide geometric quantities (area, volume, ...) and surface averaging. Obtained via `Equilibrium.flux_surface(...)`, not constructed directly.
- **`FluxFunctions` / `SurfaceFunctions`** (`fluxfunctions.py`, `surfacefunctions.py`) — containers (available as `eq.fluxfuncs` / `eq.surfacefuncs`) onto which users dynamically attach interpolated 1D flux functions or 2D surface functions via `add_flux_func` / `add_surface_func`.
- **COCOS** (`cocos.py`) — tokamak coordinate-convention coefficients per Sauter & Medvedev. COCOS support is only partially implemented; COCOS 3 is the working default assumption. Be careful with signs.

### I/O (`pleque/io/`)

Equilibria should be created through this package rather than by calling `Equilibrium(...)` directly. `readers.read_geqdsk` is the main entry point; `geqdsk.py` provides read/write, `compass.py` (CDB/FIESTA), `jet/`, `metis.py` and `omas.py` cover machine/format-specific access. Modules prefixed with `_` (`_geqdsk.py`, `_readgeqdsk.py`, ...) are internal helpers. `tools.py` provides `EquilibriaTimeSlices` for time-resolved data.

### Supporting packages

- `pleque/utils/` — numerical machinery used by `Equilibrium`: critical-point search (`equi_tools.py`), surface tracking (`surfaces.py`), field-line tracing integrators (`field_line_tracers.py`), flux expansion (`flux_expansions.py`), plotting helpers (`plotting.py`).
- `pleque/utils/decorators.py` — `@scalar_function`, `@vector_function(ndim)`, `@ordered_path_scalar_function` register `Equilibrium` methods for shape-contract testing; `@deprecated` marks deprecated API.
- `pleque/config/settings.py` — `pydantic-settings` based configuration (grid resolutions etc.), accessed via the cached `get_settings()`.
- `pleque/spatran/` — affine transforms and reference frames.

## Key Conventions

### Array shape convention (from README — enforced by `tests/test_dimenstions.py`)

- Scalar functions at paired points return `[n_elements]`; on a grid return `[n_z, n_r]` matching `np.meshgrid(R, Z)` (note: SciPy splines return `(n_R, n_Z)`, so grid results are transposed via `Equilibrium._shape_spline_result`).
- Vector functions return `[n_dim, ...]` (component-first): `[n_dim, n_elements]` for points, `[n_dim, n_z, n_r]` for grids.
- Mesh-shaped `R`/`Z` with `grid=False` means elementwise evaluation, preserving the mesh shape.

### Naming (see `docs/source/naming_convention.rst`)

`R`/`Z` cylindrical coordinates; `psi` (Wb), `psi_n` normalized flux, `rho = sqrt(psi_n)`; `B_R`, `B_Z`, `B_pol`, `B_tor`, `B_abs`; `j_*` current densities; profiles `pressure`, `pprime`, `F`, `FFprime` (with lowercase `f = F/mu_0` variants); `q` safety factor. These names are used both as method names and as dict/xarray keys in I/O.

### Versioning and releases

- The version string lives in `pleque/__init__.py` (`__version__`); the minor version is duplicated in `docs/source/conf.py`. Note `pyproject.toml` may lag behind.
- Development happens on the `develop` branch; releases go through `release/<version>` branches, are tagged `v<version>`, and published with `poetry build` + `twine upload` — see `how_to_publish_release.md`.
- The project is pre-0.1.0: breaking changes are expected, but versions 0.0.9/0.0.10 intentionally maintain back-compatibility with 0.0.8 where possible.

### Misc

- `examples/` and `notebooks/` contain usage examples that double as informal documentation; docs notebooks are linked via `.nblink` files in `docs/source/`.
- CI is GitLab CI (`.gitlab-ci.yml`), not GitHub Actions, and only runs `pytest`.
