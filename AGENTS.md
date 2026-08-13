# Repository Guidelines

## Project Structure & Module Organization

`pleque/` contains the installable Python package. Core equilibrium and coordinate behavior lives in `pleque/core/`; format readers and writers are in `pleque/io/`; shared numerical and plotting helpers are in `pleque/utils/`; spatial transformations are in `pleque/spatran/`. Keep bundled sample equilibria in `pleque/resources/`. The main test suite is under `tests/`, with reusable package-side test helpers in `pleque/tests/`. User-facing demonstrations belong in `examples/` or `notebooks/`, while Sphinx sources live in `docs/source/`.

## Build, Test, and Development Commands

- `uv sync` creates the locked development environment from `pyproject.toml` and `uv.lock`.
- `uv sync --group compass` additionally installs the optional COMPASS/CDB integration.
- `uv run pytest` runs the complete test suite configured by `pytest.ini`.
- `uv run pytest tests/test_gfile.py -q` runs a focused module while developing.
- `uv run ruff check pleque tests` checks Python style and common correctness issues.
- `uv run ty check pleque` runs the advisory type checker used by CI.
- `uv build` creates source and wheel distributions through Hatchling.
- `uv run make -C docs html` builds the Sphinx documentation into `docs/build/html/`.

## Coding Style & Naming Conventions

Use four-space indentation and follow the existing scientific Python style. Name modules, functions, and variables with `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Preserve established domain notation such as `R`, `Z`, `psi_n`, and `COCOS` where it improves correspondence with equations and file formats. Keep public APIs documented with concise docstrings, and avoid mixing unrelated formatting changes into functional patches. Use Ruff for linting and `uv run ruff format pleque tests` for automatic formatting.

## Testing Guidelines

Tests use Pytest and NumPy assertion helpers. Add tests close to the affected behavior, normally as `tests/test_<topic>.py` with functions named `test_<behavior>`. Reuse fixtures from `tests/conftest.py` and sample data from `pleque/resources/` instead of adding large generated artifacts. Cover numerical results with explicit tolerances and include regression cases for bug fixes. Run focused tests first, then the full suite before submitting.

## Commit & Pull Request Guidelines

Recent history favors short, imperative or descriptive subjects such as `Add reference publication to README.md` and `Update dependencies`. Keep each commit scoped to one logical change; use `Bump version to X.Y.Z` for release-only version commits. Pull requests should explain the user-visible or numerical impact, list verification commands, and link relevant issues. Include plots or screenshots when visualization output changes, and call out compatibility, dependency, or equilibrium-data changes explicitly.
