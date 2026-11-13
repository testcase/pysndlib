"""Tests for the store-and-raise sndlib error handler model."""
import numpy as np
import pytest

import pysndlib.clm as clm


def test_missing_file_error_surfaces_at_boundary(tmp_path):
    clm.get_srate(str(tmp_path / "definitely_missing.wav"))
    with pytest.raises(clm.SNDLibError):
        clm.check_sndlib_errors()


def test_slot_is_clean_after_raise(tmp_path):
    clm.get_srate(str(tmp_path / "definitely_missing.wav"))
    with pytest.raises(clm.SNDLibError):
        clm.check_sndlib_errors()
    # a second check must not raise: the error was consumed
    clm.check_sndlib_errors()


def test_from_ptr_surfaces_error_from_make():
    # a C-level failure inside mus_make_* stores the error and from_ptr
    # raises it at the boundary
    with pytest.raises(Exception):
        clm.make_delay(-1)
    clm.check_sndlib_errors()


def test_make_env_rejects_bad_envelope_python_side():
    with pytest.raises(RuntimeError):
        clm.make_env("not an envelope", duration=0.01)
    clm.check_sndlib_errors()


def test_ndarray2file_bad_output_dir_raises(tmp_path):
    target = str(tmp_path / "no" / "such" / "dir" / "out.wav")
    with pytest.raises(clm.SNDLibError):
        clm.ndarray2file(target, np.zeros((1, 10)), sr=44100)
    clm.check_sndlib_errors()  # slot must be clean afterwards


def test_check_sndlib_errors_is_noop_when_clean():
    clm.check_sndlib_errors()  # must not raise


def test_validate_envelope_rejects_short_envelopes():
    with pytest.raises(RuntimeError, match="at least 2"):
        clm.validate_envelope([0])
    with pytest.raises(RuntimeError, match="even number"):
        clm.validate_envelope([0, 0, 100])


def test_validate_envelope_rejects_non_numeric():
    with pytest.raises(RuntimeError, match="only contain numbers"):
        clm.validate_envelope([0, "a", 100, "b"])


def test_validate_envelope_rejects_non_increasing_x():
    with pytest.raises(RuntimeError, match="increasing"):
        clm.validate_envelope([0, 0, 50, 1, 25, 0])


def test_validate_envelope_accepts_valid():
    assert clm.validate_envelope([0, 0, 50, 1, 100, 0]) is None


def test_sndlib_error_is_raisable_with_message():
    err = clm.SNDLibError("boom")
    assert "boom" in str(err)
    with pytest.raises(clm.SNDLibError, match="boom"):
        raise err
