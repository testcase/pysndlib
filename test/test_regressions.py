"""Regression pins for the bugs fixed in the 0.8.0 modernization.

Each test cross-references CHANGES-0.8.0.md. If one of these fails after a
future change, the corresponding behavior was broken.
"""
import os

import numpy as np
import pytest


import pysndlib.clm as clm
import pysndlib.generators as gens
import pysndlib.sndlib as sndlib


def test_is_odd_and_is_even_not_swapped():
    # CHANGES-0.8.0: is_odd/is_even were swapped
    assert clm.is_odd(7) and not clm.is_odd(8)
    assert clm.is_even(8) and not clm.is_even(7)


def test_in_any_tests_parameter_not_builtin_input():
    # CHANGES-0.8.0: in_any checked the builtin input, making the ndarray
    # branch unreachable
    inp = np.arange(20, dtype=np.double).reshape((2, 10))
    assert clm.in_any(5, 1, inp) == 15.0


def test_mus_sound_override_header_passes_filename(tmp_path):
    # CHANGES-0.8.0: override_header passed the type object `str` instead of
    # the filename
    path = str(tmp_path / "over.wav")
    clm.ndarray2file(path, np.zeros((1, 10)), sr=44100)
    sample_type = sndlib.mus_sound_sample_type(path)
    header_type = sndlib.mus_sound_header_type(path)
    assert sndlib.mus_sound_override_header(
        path, 44100, 1, sample_type, header_type, 0, 10) == 0


def test_locsig_returns_none():
    # CHANGES-0.8.0: locsig was declared cpdef double with no return
    out = np.zeros((1, 4))
    with clm.Sound(out, srate=44100):
        loc = clm.make_locsig(0.0, 1.0, 0.0)
        assert clm.locsig(loc, 0, 0.5) is None
    assert out[0][0] == pytest.approx(0.5)


def test_partials2wave_accepts_explicit_ndarray_wave():
    # CHANGES-0.8.0: `if not wave:` raised ValueError for ndarray wave input
    wave = np.zeros(512)
    result = clm.partials2wave([1, 1], wave=wave)
    assert result is wave
    assert float(np.abs(result).max()) > 0.5


def test_partials2wave_allocates_when_wave_is_none():
    result = clm.partials2wave([1, 1])
    assert isinstance(result, np.ndarray)
    assert float(np.abs(result).max()) == pytest.approx(1.0, abs=1e-6)


def test_make_fir_coeffs_accepts_list_envelope():
    # CHANGES-0.8.0: np.ndarray(envelope) raised for list input
    coeffs = clm.make_fir_coeffs(8, [0, 0, 50, 1, 100, 0])
    assert isinstance(coeffs, np.ndarray)
    assert np.all(np.isfinite(coeffs))


def test_mus_sound_reopen_output_has_no_srate_parameter():
    # CHANGES-0.8.0 (API change): srate was never passed to sndlib
    import inspect
    sig = inspect.signature(sndlib.mus_sound_reopen_output)
    assert "srate" not in sig.parameters


def test_validate_envelope_logic_not_inverted():
    # CHANGES-0.8.0: the "only contain numbers" check was inverted and the
    # message was not an f-string
    with pytest.raises(RuntimeError, match="7"):  # odd length reported in message
        clm.validate_envelope([0, 0, 1, 1, 2, 3, 4])


def test_sndlib_error_has_working_exception_behavior():
    # CHANGES-0.8.0: the broken triple-underscore ___init___ was removed
    with pytest.raises(clm.SNDLibError):
        raise clm.SNDLibError("message")


def test_moving_fft_is_instantiable(sine_wav):
    # CHANGES-0.8.0: attribute block was commented out in generators.pxd
    path, sr, nsamples, freq = sine_wav
    reader = clm.make_readin(path)
    mf = gens.make_moving_fft(reader, 256, 64)
    assert isinstance(mf, gens.CLMGenerator)
    assert mf.n == 256 and mf.hop == 64


def test_nchoosekcos_signature_is_frequency_ratio_n():
    # CHANGES-0.8.0 (API change): factory arg order was fixed
    gen = gens.make_nchoosekcos(440.0, ratio=0.5, n=4)
    assert gen.ratio == 0.5
    assert gen.n == 4


def test_roundinterp_mus_phase_setter():
    # CHANGES-0.8.0: setter targeted nonexistent self.gen.rnd
    gen = gens.make_round_interp(440.0, 4)
    gen.mus_phase = 0.25
    assert gen.mus_phase == pytest.approx(0.25)


def test_mus_sound_loop_info_returns_owned_array(sine_wav):
    # CHANGES-0.8.0: loop_info leaked sndlib's buffer; now returns an owned
    # array (None here since the file has no loop info)
    path, sr, nsamples, freq = sine_wav
    info = sndlib.mus_sound_loop_info(path)
    assert info is None or (isinstance(info, np.ndarray) and len(info) == 8)


def test_ndarray2file_handles_1d_input(tmp_path):
    # 1-D arrays are reshaped to (1, n) before writing
    path = str(tmp_path / "one_d.wav")
    clm.ndarray2file(path, np.zeros(16) + 0.5, sr=44100)
    arr, _ = clm.file2ndarray(path)
    assert arr.shape == (1, 16)
    assert np.all(arr == 0.5)


def test_envelope_interp_accepts_lists():
    # found by the pytest suite: envelope_interp only accepted env generators,
    # but CLM's env-interp (and canter / fm_voice) evaluate breakpoint lists
    assert clm.envelope_interp(50, [0, 200, 100, 600]) == pytest.approx(400.0)
    assert clm.envelope_interp(25, [0, 200, 50, 400, 100, 600]) == pytest.approx(300.0)
    # clamping outside the breakpoint range
    assert clm.envelope_interp(-5, [0, 1, 100, 2]) == pytest.approx(1.0)
    assert clm.envelope_interp(150, [0, 1, 100, 2]) == pytest.approx(2.0)


def test_noddsin_is_instantiable():
    # found by the pytest generator sweep: Noddsin's attribute block was
    # missing from generators.pxd, so no Noddsin could ever be constructed
    gen = gens.make_noddsin(440.0, 1.0, 3)
    assert gens.is_noddsin(gen)
    import math
    assert math.isfinite(gen.next())

