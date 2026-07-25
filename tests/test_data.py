import importlib
import os

from flight_routes import data


def test_raw_dir_picks_up_env_var_set_after_import(monkeypatch):
    # Regression test: in the notebook, `import flight_routes` (cell 0) happens
    # before `os.environ['FLIGHT_ROUTES_RAW_DIR'] = ...` (cell 1, after Drive
    # mount). raw_dir()/cache_dir() must read the env var fresh on every call,
    # not freeze it as a module-level constant at import time.
    monkeypatch.setenv("FLIGHT_ROUTES_RAW_DIR", "/drive/MyDrive/flight-project")
    assert data.raw_dir().as_posix() == "/drive/MyDrive/flight-project"


def test_raw_dir_default_when_unset(monkeypatch):
    monkeypatch.delenv("FLIGHT_ROUTES_RAW_DIR", raising=False)
    assert str(data.raw_dir()).endswith(os.path.join("data", "raw"))


def test_cache_dir_default_when_unset(monkeypatch):
    monkeypatch.delenv("FLIGHT_ROUTES_CACHE_DIR", raising=False)
    assert str(data.cache_dir()).endswith(os.path.join("data", "processed"))
