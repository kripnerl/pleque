# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Shared contributor instructions are defined in `AGENTS.md`; this file adds Claude-specific project context.

## Overview

PLEQUE (**PL**asma **EQU**ilibrium **E**njoyment module) is a Python library for visualisation and
manipulation of tokamak plasma equilibria. Its central abstraction is the `Equilibrium` class,
typically constructed by reading an equilibrium file (e.g. G-EQDSK) via `pleque.io.readers`.

## Commands

The project uses uv. The `compass` dependency group pulls `pycdb-compass` from an internal
IPP git repository that is not reachable outside the COMPASS network, so install it only when needed:

```bash
uv sync                                      # install locked default and development dependencies
uv sync --group compass                      # additionally install the COMPASS integration

uv run pytest                                # run the whole test suite
uv run pytest tests/test_equilibria.py       # run one test file
uv run pytest tests/test_equilibria.py::test_name       # run a single test

uv run ruff check pleque/ tests/              # lint (CI runs exactly this)
uv run ruff format pleque/ tests/             # format (line-length 120, double quotes)

uv run ty check pleque/                       # type check (advisory: CI runs it with continue-on-error
                                              # because the legacy codebase is largely un-annotated)
```

Tests that need unavailable optional dependencies (e.g. `pyCDB` in `tests/test_cdb.py`) skip
themselves via `pytest.importorskip` — they are not failures.

Docs are Sphinx-based in `docs/` (built on Read the Docs): `uv run make -C docs html`.

## Branches and releases

- `develop` is the main development branch; `master` holds releases.
- CI (`.github/workflows/ci.yml`) runs tests on Python 3.10 and 3.11 for pushes to `master`,
  `develop`, and `claude/**` branches, then lint (ruff) and type check (ty).
- The version string lives in `pleque/__init__.py` (`__version__`), with the minor version also
  in `docs/source/conf.py`. The release process is documented in `how_to_publish_release.md`.

## Architecture

### Core (`pleque/core/`)

- `equilibrium.py` — `Equilibrium`, the heart of the package (~2000 lines). Wraps an
  `xarray.Dataset` of poloidal flux on an (R, Z) grid plus 1D profiles, builds 2D/1D splines
  (`RectBivariateSpline` / `UnivariateSpline`), finds the magnetic axis, X-points, LCFS and
  strike points, and exposes evaluation methods (`B_R`, `B_Z`, `B_tor`, `B_abs`, `psi`, `q`, ...),
  flux-surface generation, mapping, and plotting helpers.
- `coordinates.py` — `Coordinates`, the universal coordinate container. Nearly every public
  `Equilibrium` method accepts flexible coordinate input (1D `psi_n`, 2D `(R, Z)`, 3D
  `(R, Z, phi)`, arrays, grids, other `Coordinates`) and normalises it through this class.
- `fluxsurface.py` — `Surface` / `FluxSurface` (closed/open contours, built on `shapely`),
  with geometry properties (area, volume, geometric averages).
- `fluxfunctions.py` / `surfacefunctions.py` — containers for 1D profile functions attached to
  an equilibrium.
- `cocos.py` — COCOS (tokamak COordinate COnventionS, Sauter & Medvedev) coefficient handling.
  Readers take a `cocos` argument; `Coordinates` carries cocos through transformations.

**Import-order caveat:** `pleque/core/__init__.py` has a load-order-dependent import sequence
due to circular imports (`Equilibrium` must be imported last). Ruff `I001`/`F401` are suppressed
for it in `pyproject.toml` — do not let an auto-formatter reorder those imports.

### IO (`pleque/io/`)

- `readers.py` — high-level entry points (`read_geqdsk`, ...) that return `Equilibrium` instances.
- `geqdsk.py` is the public G-EQDSK read/write API; `_geqdsk.py` contains the low-level format
  routines (vendored from FreeGS, LGPL — keep its license header intact).
- Per-source readers: `compass.py` (CDB / FIESTA / EFIT HDF5), `omas.py` (OMAS/IMAS),
  `metis.py`, `jet/`. Optional heavy dependencies are imported inside functions so the package
  works without them.
- `io/__init__.py` files re-export the public API (`F401` suppressed for them).

### Supporting packages

- `pleque/utils/` — numerics behind `Equilibrium`: `surfaces.py` (contour finding, boundary
  tracking), `field_line_tracers.py`, `flux_expansions.py`, `equi_tools.py`, `plotting.py`, and
  `decorators.py` (see array convention below).
- `pleque/config/settings.py` — `pydantic-settings` based `Settings`; access via the
  `lru_cache`d `get_settings()`, so settings are effectively process-global.
- `pleque/spatran/` — affine transformations and reference frames for spatial transforms.
- `pleque/resources/` — bundled test equilibria (eqdsk/gfile/netCDF files).
- `pleque/tests/utils.py` — loads the bundled equilibria (`load_testing_equilibrium(case)`);
  used by `tests/conftest.py`, whose module-scoped `equilibrium` fixture parametrizes most tests
  over six bundled test cases.

### Array convention (important for any new evaluation function)

Public evaluation functions follow a component-first convention, enforced via decorators in
`pleque/utils/decorators.py` (`scalar_function`, `vector_function`,
`ordered_path_scalar_function`):

- Scalars at paired points return `[n_elements]`; on a grid (`grid=True`) return `[n_z, n_r]`,
  matching `np.meshgrid(R, Z)`.
- Vectors return `[n_dim, ...]` (e.g. `[n_dim, n_z, n_r]` for grids).
- SciPy's `RectBivariateSpline` returns grids as `(n_R, n_Z)`;
  `Equilibrium._shape_spline_result` transposes to the public `(n_Z, n_R)` layout — use it when
  adding spline-backed methods.
