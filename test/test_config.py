"""Tests for the pysndlib settings (rc) file loading."""
import pytest

import pysndlib.clm as clm


def test_load_rc_explicit_path(tmp_path):
    rc = tmp_path / "rc.py"
    rc.write_text("default.srate = 48000\n")
    old = clm.default.srate
    try:
        assert clm.load_rc(str(rc)) is True
        assert clm.default.srate == 48000
    finally:
        clm.default.srate = old


def test_load_rc_bare_assignment_persists_in_module(tmp_path):
    # back-compat: the historical module-level exec landed bare assignments
    # in the clm module namespace; load_rc must keep doing that
    rc = tmp_path / "rc_bare.py"
    rc.write_text("pysndlib_rc_test_marker = 42\n")
    try:
        assert clm.load_rc(str(rc)) is True
        assert clm.pysndlib_rc_test_marker == 42
    finally:
        if hasattr(clm, "pysndlib_rc_test_marker"):
            del clm.pysndlib_rc_test_marker


def test_load_rc_missing_file_returns_false(tmp_path):
    assert clm.load_rc(str(tmp_path / "nope.py")) is False


def test_load_rc_broken_file_warns(tmp_path):
    rc = tmp_path / "broken.py"
    rc.write_text("this is not valid python ((\n")
    with pytest.warns(UserWarning, match="error loading pysndlib settings"):
        assert clm.load_rc(str(rc)) is True


def test_load_rc_strict_mode_raises(tmp_path, monkeypatch):
    rc = tmp_path / "raises.py"
    rc.write_text("raise ValueError('boom')\n")
    monkeypatch.setenv("PYSNDLIB_RC_STRICT", "1")
    with pytest.raises(ValueError, match="boom"):
        clm.load_rc(str(rc))


def test_rc_paths_env_override(monkeypatch):
    monkeypatch.setenv("PYSNDLIB_RC", "/some/override.py")
    assert clm._rc_paths() == ["/some/override.py"]


def test_rc_paths_platform_location_and_legacy_fallback(monkeypatch):
    import os
    import sys
    # clear any outer override (e.g. a CI runner isolating imports via PYSNDLIB_RC)
    monkeypatch.delenv("PYSNDLIB_RC", raising=False)
    paths = clm._rc_paths()
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
    elif sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", os.path.join(home, ".config"))
    assert paths[0] == os.path.join(base, "pysndlib", "pysndlib.py")
    assert paths[1] == os.path.join(home, ".pysndlib")  # legacy fallback
