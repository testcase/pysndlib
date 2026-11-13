"""Tests for pysndlib.sndlib — the sound-file / header API."""
import os

import numpy as np
import pytest

import pysndlib.clm as clm
import pysndlib.sndlib as sndlib


def test_sound_file_header_info(sine_wav):
    path, sr, nsamples, freq = sine_wav
    assert sndlib.mus_sound_samples(path) == nsamples
    assert sndlib.mus_sound_framples(path) == nsamples
    assert sndlib.mus_sound_srate(path) == sr
    assert sndlib.mus_sound_chans(path) == 1
    assert sndlib.mus_sound_duration(path) == pytest.approx(nsamples / sr, rel=1e-3)
    assert isinstance(sndlib.mus_sound_header_type(path), sndlib.Header)
    assert isinstance(sndlib.mus_sound_sample_type(path), sndlib.Sample)


def test_header_type_matches_sound_default(sine_wav):
    path, sr, nsamples, freq = sine_wav
    header = sndlib.mus_sound_header_type(path)
    assert header == clm.default.header_type
    name = sndlib.mus_header_type_to_string(header)
    assert isinstance(name, str) and len(name) > 0


def test_sound_comment_is_a_string_or_none(sine_wav):
    path, sr, nsamples, freq = sine_wav
    comment = sndlib.mus_sound_comment(path)
    # files without a comment return None (0.8.0: this used to segfault on
    # the NULL char* conversion)
    assert comment is None or isinstance(comment, str)


def test_loop_info_without_loop_data_is_none(sine_wav):
    path, sr, nsamples, freq = sine_wav
    info = sndlib.mus_sound_loop_info(path)
    assert info is None or (isinstance(info, np.ndarray) and len(info) == 8)


def test_open_and_close_input(sine_wav):
    path, sr, nsamples, freq = sine_wav
    fd = sndlib.mus_sound_open_input(path)
    assert isinstance(fd, int) and fd != -1
    assert sndlib.mus_sound_close_input(fd) == 0


def test_open_input_missing_file_returns_minus_one(tmp_path):
    fd = sndlib.mus_sound_open_input(str(tmp_path / "missing.wav"))
    assert fd == -1


def test_override_header_updates_cached_info(sine_wav):
    path, sr, nsamples, freq = sine_wav
    sample_type = sndlib.mus_sound_sample_type(path)
    header_type = sndlib.mus_sound_header_type(path)
    result = sndlib.mus_sound_override_header(
        path, 22050, 1, sample_type, header_type, 0, nsamples)
    assert result == 0
    assert sndlib.mus_sound_srate(path) == 22050


def test_reopen_output_signature(sine_wav, tmp_path):
    """mus_sound_reopen_output takes no srate parameter (removed in 0.8.0)."""
    path, sr, nsamples, freq = sine_wav
    sample_type = sndlib.mus_sound_sample_type(path)
    header_type = sndlib.mus_sound_header_type(path)
    out = str(tmp_path / "reopen.wav")
    arr = np.zeros((1, 10))
    clm.ndarray2file(out, arr, sr=sr)
    fd = sndlib.mus_sound_open_output(out, sr, 1, sample_type, header_type, "")
    assert fd != -1
    sndlib.mus_sound_close_output(fd, 0)
    result = sndlib.mus_sound_reopen_output(out, 1, sample_type, header_type, 4)
    assert isinstance(result, int)


def test_mus_file_round_trip(sine_wav):
    """The low-level fd API reads back the same samples sndlib wrote."""
    path, sr, nsamples, freq = sine_wav
    fd = sndlib.mus_sound_open_input(path)
    assert fd != -1
    assert sndlib.mus_sound_close_input(fd) == 0
    data, _ = clm.file2ndarray(path)
    assert np.all(np.isfinite(data))

