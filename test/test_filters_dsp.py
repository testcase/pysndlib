"""Tests for clm filters and pysndlib.dsp."""
import os

import numpy as np
import pytest

import pysndlib.clm as clm
import pysndlib.dsp as dsp



# ---- FIR filter helpers from dsp ----

@pytest.mark.parametrize("factory", [dsp.make_highpass, dsp.make_lowpass])
def test_fir_highlow_pass_constructs_and_runs(factory):
    gen = factory(0.2)
    for _ in range(50):
        v = gen(0.5)
        assert np.isfinite(v)


def test_bandpass_and_bandstop_run():
    for gen in (dsp.make_bandpass(0.1, 0.3), dsp.make_bandstop(0.1, 0.3)):
        for _ in range(50):
            assert np.isfinite(gen(0.25))


def test_dsp_wrappers_match_clm_filters():
    h1, h2 = dsp.make_highpass(0.2), dsp.make_highpass(0.2)
    assert dsp.highpass(h1, 0.5) == pytest.approx(clm.fir_filter(h2, 0.5))


# ---- biquad ----

def test_biquad_returns_filter_output():
    """Regression (0.8.0): biquad() was missing its return statement."""
    gen = dsp.make_biquad(1.0, 0.0, 0.0, 0.0, 0.0)
    assert dsp.biquad(gen, 0.5) == pytest.approx(0.5)


def test_biquad_iir_constructions():
    lp = dsp.make_iir_low_pass_2(1000.0, 0.707)
    bp = dsp.make_iir_band_pass_2(400.0, 800.0)
    bs = dsp.make_iir_band_stop_2(400.0, 800.0)
    for gen in (lp, bp, bs):
        for _ in range(50):
            assert np.isfinite(clm.filter(gen, 0.25))


def test_clm_biquad_variants_construct():
    for gen in (clm.make_biquad_notch(440.0, 0.9),
                clm.make_biquad_low_pass(440.0, 1.0),
                clm.make_biquad_high_pass(440.0, 1.0),
                clm.make_biquad_band_pass(440.0, 1.0),
                clm.make_biquad_band_reject(440.0, 1.0),
                clm.make_biquad_all_pass(440.0, 1.0)):
        for _ in range(20):
            assert np.isfinite(gen(0.5))


def test_butterworth_filters():
    for gen in (dsp.make_butter_high_pass(1000.0),
                dsp.make_butter_low_pass(1000.0),
                dsp.make_butter_band_pass(1000.0, 200.0),
                dsp.make_butter_band_reject(1000.0, 200.0)):
        for _ in range(50):
            assert np.isfinite(dsp.butter(gen, 0.25))


# ---- filter behavior checks ----

def test_one_pole_lowpass_converges_to_dc():
    # sndlib convention: y(n) = a0*x(n) - b1*y(n-1); DC gain = a0/(1+b1)
    gen = clm.make_one_pole(0.5, -0.5)  # DC gain 1.0
    out = [clm.one_pole(gen, 1.0) for _ in range(500)]
    assert out[-1] == pytest.approx(1.0, abs=1e-9)


def test_00_two_pole_isolation_probe():
    """Position-0 probe: does the two_pole ycoeffs path work BEFORE any other
    test runs? (cross-test corruption discriminator for the Windows cp311 bug)"""
    gen = clm.make_two_pole(a0=1.0, b1=0.5, b2=0.25)
    ycoeffs = np.asarray(gen.mus_ycoeffs)
    assert ycoeffs[1] == pytest.approx(0.5)
    assert ycoeffs[2] == pytest.approx(0.25)


def test_two_pole_uses_b1_not_b2_twice():
    """Regression (0.8.0): make_two_pole passed (a0, b2, b2)."""
    gen = clm.make_two_pole(a0=1.0, b1=0.5, b2=0.25)
    try:
        ycoeffs = np.asarray(gen.mus_ycoeffs)
    except TypeError:
        # --- diagnostics for the Windows cp311 release-blocker failure ---
        print("=== two_pole failure diagnostics ===")
        print("is_two_pole   :", clm.is_two_pole(gen))
        print("str(gen)      :", str(gen))
        for label, fn in [("mus_xcoeffs", lambda: gen.mus_xcoeffs),
                          ("mus_length  ", lambda: gen.mus_length),
                          ("two_pole run", lambda: clm.two_pole(gen, 1.0)),
                          ("ycoeffs retry", lambda: gen.mus_ycoeffs)]:
            try:
                print(label, ":", fn())
            except Exception as e:
                print(label, ": raised", repr(e))
        print("=== end diagnostics ===")
        raise
    assert ycoeffs[1] == pytest.approx(0.5)
    assert ycoeffs[2] == pytest.approx(0.25)


# ---- src helpers ----

def test_src_duration_known_value():
    assert dsp.src_duration([0, 1, 1, 1]) == pytest.approx(1.0)


def test_src_fit_envelope_preserves_length():
    e = [0, 1, 50, 2, 100, 1]
    out = dsp.src_fit_envelope(e, 0.5)
    assert len(out) == len(e)


# ---- file-based dsp ----

def test_freqdiv_repeats_blocks(sine_wav, tmp_path):
    path, sr, nsamples, freq = sine_wav
    out = str(tmp_path / "freqdiv.wav")
    dsp.freqdiv(4, path, 0, out)
    data, _ = clm.file2ndarray(out)
    assert data.shape == (1, nsamples)
    # every group of n samples should be constant
    for i in range(0, 24, 4):
        group = data[0][i:i + 4]
        assert np.all(group == group[0])


def test_freqdiv_requires_outname(sine_wav):
    path, sr, nsamples, freq = sine_wav
    # regression (0.8.0): validation was unreachable when n <= 1
    with pytest.raises(ValueError):
        dsp.freqdiv(1, path, 0, None)


def test_down_oct_returns_existing_file(sine_wav, tmp_path):
    """Regression (0.8.0): down_oct returned a filename it never wrote."""
    path, sr, nsamples, freq = sine_wav
    out = str(tmp_path / "down.wav")
    returned = dsp.down_oct(2, path, 0, out)
    assert returned == out
    assert os.path.isfile(returned)
    data, _ = clm.file2ndarray(returned)
    assert np.all(np.isfinite(data))
