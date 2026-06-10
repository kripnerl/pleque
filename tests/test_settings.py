"""Tests of the PLEQUE settings module: defaults, file loading precedence, source tracking."""

import os

import pytest
from pydantic import ValidationError

from pleque.config import settings as settings_module
from pleque.config.settings import (
    CONFIG_FILE_ENV_VAR,
    CONFIG_FILE_NAME,
    Settings,
    get_settings,
    load_settings,
    reload_settings,
    user_config_path,
)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Run each test in an empty cwd with no PLEQUE env vars and a cold settings cache."""
    for key in list(os.environ):
        if key.startswith("PLEQUE_"):
            monkeypatch.delenv(key)
    # point the user config dir somewhere empty so a real ~/.config/pleque does not leak in
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_without_any_config(tmp_path):
    settings = get_settings()
    assert settings.flux_surfaces.n_psi == 200
    assert settings.flux_surfaces.psi_n_min == 0.01
    assert settings.lcfs.search_grid_nr == 700
    assert settings.lcfs.search_grid_nz == 1200
    assert settings.default_cocos == 3
    assert settings.config_source is None
    assert tmp_path / CONFIG_FILE_NAME in settings.config_files_consulted


def test_load_from_cwd_toml(tmp_path):
    (tmp_path / CONFIG_FILE_NAME).write_text("[flux_surfaces]\nn_psi = 321\n")
    settings = get_settings()
    assert settings.flux_surfaces.n_psi == 321
    # untouched values keep their defaults
    assert settings.lcfs.search_grid_nr == 700
    assert settings.config_source == tmp_path / CONFIG_FILE_NAME


def test_cwd_beats_user_config_dir(tmp_path):
    user_file = user_config_path()
    user_file.parent.mkdir(parents=True)
    user_file.write_text("[flux_surfaces]\nn_psi = 111\n\n[lcfs]\nsearch_grid_nr = 999\n")
    (tmp_path / CONFIG_FILE_NAME).write_text("[flux_surfaces]\nn_psi = 222\n")

    settings = get_settings()
    assert settings.flux_surfaces.n_psi == 222
    assert settings.config_source == tmp_path / CONFIG_FILE_NAME
    # files are not merged: the loser file is ignored completely
    assert settings.lcfs.search_grid_nr == 700


def test_user_config_dir_posix(tmp_path):
    user_file = user_config_path()
    assert user_file == tmp_path / "xdg" / "pleque" / CONFIG_FILE_NAME
    user_file.parent.mkdir(parents=True)
    user_file.write_text("[grid]\ndefault_nr = 555\n")

    settings = get_settings()
    assert settings.grid.default_nr == 555
    assert settings.config_source == user_file


def test_user_config_dir_windows(tmp_path, monkeypatch):
    appdata = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(appdata))
    assert settings_module._windows_config_dir() == appdata / "pleque"

    # without APPDATA the home directory is used as a fallback
    monkeypatch.delenv("APPDATA")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert settings_module._windows_config_dir() == tmp_path / "home" / "pleque"


def test_pyproject_tool_pleque(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[tool.pleque.grid]\ndefault_nr = 555\n")
    subdir = tmp_path / "some" / "subdir"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(subdir)

    settings = get_settings()
    assert settings.grid.default_nr == 555
    assert settings.config_source == tmp_path / "pyproject.toml"


def test_pyproject_without_table_ignored(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.other]\nfoo = 1\n")
    settings = get_settings()
    assert settings.grid.default_nr == 1000
    assert settings.config_source is None


def test_explicit_config_file_env_var(tmp_path, monkeypatch):
    (tmp_path / CONFIG_FILE_NAME).write_text("[flux_surfaces]\nn_psi = 111\n")
    explicit = tmp_path / "elsewhere" / "my_config.toml"
    explicit.parent.mkdir()
    explicit.write_text("[flux_surfaces]\nn_psi = 333\n")
    monkeypatch.setenv(CONFIG_FILE_ENV_VAR, str(explicit))

    settings = get_settings()
    assert settings.flux_surfaces.n_psi == 333
    assert settings.config_source == explicit
    assert settings.config_files_consulted == (explicit,)


def test_explicit_config_file_env_var_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_FILE_ENV_VAR, str(tmp_path / "does_not_exist.toml"))
    with pytest.raises(FileNotFoundError):
        load_settings()


def test_env_overrides_file(tmp_path, monkeypatch):
    (tmp_path / CONFIG_FILE_NAME).write_text("[flux_surfaces]\nn_psi = 111\n")
    monkeypatch.setenv("PLEQUE_FLUX_SURFACES__N_PSI", "999")

    settings = get_settings()
    assert settings.flux_surfaces.n_psi == 999
    # the file is still recorded as the source of the remaining values
    assert settings.config_source == tmp_path / CONFIG_FILE_NAME


def test_nested_env_var_types(monkeypatch):
    monkeypatch.setenv("PLEQUE_LCFS__REFINEMENT_TOLERANCE", "1e-8")
    settings = get_settings()
    assert settings.lcfs.refinement_tolerance == pytest.approx(1e-8)

    monkeypatch.setenv("PLEQUE_LCFS__REFINEMENT_TOLERANCE", "not-a-number")
    with pytest.raises(ValidationError):
        load_settings()


def test_init_kwargs_override_everything(tmp_path):
    (tmp_path / CONFIG_FILE_NAME).write_text("[flux_surfaces]\nn_psi = 111\n")
    settings = Settings(flux_surfaces={"n_psi": 42})
    assert settings.flux_surfaces.n_psi == 42


def test_reload_settings(tmp_path):
    assert get_settings().flux_surfaces.n_psi == 200
    (tmp_path / CONFIG_FILE_NAME).write_text("[flux_surfaces]\nn_psi = 321\n")

    settings = reload_settings()
    assert settings.flux_surfaces.n_psi == 321
    assert get_settings() is settings


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
    assert load_settings() is not get_settings()


def test_no_import_time_settings_capture():
    """Call sites must read settings at use time, not capture them at import time."""
    import pleque.core.equilibrium as equilibrium_module

    assert not hasattr(equilibrium_module, "settings")


def test_settings_used_at_call_time(tmp_path):
    """Functions resolving defaults from settings must see freshly reloaded values."""
    from pleque.utils import surfaces

    (tmp_path / CONFIG_FILE_NAME).write_text("[lcfs]\nx_point_shift = 0.125\n")
    reload_settings()
    assert settings_module.get_settings().lcfs.x_point_shift == 0.125
    # get_settings imported inside pleque.utils.surfaces is the same cached object
    assert surfaces.get_settings().lcfs.x_point_shift == 0.125


def test_docs_settings_reference_complete():
    """Every settings section and field must be documented in docs/source/configuration.rst."""
    from pathlib import Path

    docs = Path(__file__).parent.parent / "docs" / "source" / "configuration.rst"
    text = docs.read_text()

    for section_name, field in Settings.model_fields.items():
        assert section_name in text, f"settings entry '{section_name}' missing in configuration.rst"
        annotation = field.annotation
        if hasattr(annotation, "model_fields"):
            for field_name in annotation.model_fields:
                assert field_name in text, f"setting '{section_name}.{field_name}' missing in configuration.rst"
