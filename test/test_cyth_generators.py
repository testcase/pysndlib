"""Tests for the pure-Cython generator classes in pysndlib.generators."""
import math

import numpy as np
import pytest

import pysndlib.clm as clm
import pysndlib.generators as gens



# ---- K3sin: inline Horner must match mus_polynomial ----

def test_k3sin_matches_clm_polynomial():
    k3 = gens.make_k3sin(440.0)
    frq = clm.hz2radians(440.0)
    x = 0.0
    for _ in range(5):
        expected = clm.polynomial(k3.coeffs, x)
        assert k3.next() == pytest.approx(expected, abs=1e-12)
        x += frq


# ---- PinkNoise ----

def test_pink_noise_finite_and_bounded():
    pn = gens.make_pink_noise(12)
    vals = [pn.next() for _ in range(2000)]
    assert all(math.isfinite(v) for v in vals)
    assert max(np.abs(vals)) < 100.0


def test_pink_noise_is_pink_noise():
    pn = gens.make_pink_noise(6)
    assert gens.is_pink_noise(pn)


# ---- Nchoosekcos ----

def test_nchoosekcos_argument_order():
    """Regression (0.8.0): the factory passed (frequency, n, amplitude) into a
    (frequency, ratio, n) constructor."""
    gen = gens.make_nchoosekcos(440.0, 2.0, 3)
    assert gen.ratio == 2.0
    assert gen.n == 3
    assert gen.angle == 0.0  # regression (0.8.0): angle read uninitialized memory
    value = gen.next()
    assert math.isfinite(value)


# ---- SincTrain ----

def test_sinc_train_first_sample_is_one():
    """Regression (0.8.0): __init__ overwrote the widened order."""
    st = gens.make_sinc_train(100.0, 3)
    assert st.next() == pytest.approx(1.0)
    for _ in range(10):
        assert math.isfinite(st.next())


# ---- MovingFFT ----

def test_moving_fft_instantiates_and_reads(sine_wav):
    """Regression (0.8.0): the class attribute block was commented out in the
    pxd, so no MovingFFT could ever be constructed."""
    path, sr, nsamples, freq = sine_wav
    reader = clm.make_readin(path)
    mf = gens.make_moving_fft(reader, 256, 64)
    assert mf.next() == 1.0          # first call reads a full window
    assert mf.next() == 0.0           # not yet a new hop
    assert np.all(np.isfinite(mf.rl))


def test_moving_fft_mus_run_returns_flag(sine_wav):
    path, sr, nsamples, freq = sine_wav
    reader = clm.make_readin(path)
    mf = gens.make_moving_fft(reader, 256, 64)
    result = mf.mus_run(None, None)
    assert result in (True, False, 0, 1)


# ---- env connection functions ----

def test_sine_env_first_value():
    e = clm.make_env([0, 0, 100, 1])
    assert gens.sine_env(e) == pytest.approx(0.0, abs=1e-9)


def test_square_env_first_value():
    e = clm.make_env([0, 0, 100, 1])
    assert gens.square_env(e) == pytest.approx(0.0, abs=1e-9)


def test_blackman4_env_first_value():
    e = clm.make_env([0, 0, 100, 1])
    # blackman4 window evaluated at y=0
    cx = math.cos(0.0)
    expected = 0.084037 + (cx * ((cx * (0.375696 + (cx * ((cx * 0.041194) - 0.20762)))) - 0.29145))
    assert gens.blackman4_env(e) == pytest.approx(expected, abs=1e-6)


# ---- representative sweep of the cdef-class generators ----

CYTH_GENERATORS = [
    ("nssb", (440.0,)),
    ("nxysin", (440.0, 1.0, 1)),
    ("nxycos", (440.0, 1.0, 1)),
    ("nxy1cos", (440.0, 1.0)),
    ("noddsin", (440.0, 1)),
    ("noddcos", (440.0, 1)),
    ("ncos2", (440.0,)),
    ("ncos4", (440.0,)),
    ("npcos", (440.0,)),
    ("nsin5", (440.0,)),
    ("r2cos", (440.0,)),
    ("k2sin", (440.0,)),
    ("dblsum", (440.0,)),
    ("rcos", (440.0,)),
    ("rssb", (440.0,)),
]


@pytest.mark.parametrize("name", [e[0] for e in CYTH_GENERATORS])
def test_cyth_generator_constructs_and_runs(name):
    args = next(a for n, a in CYTH_GENERATORS if n == name)
    gen = getattr(gens, "make_" + name)(*args)
    assert isinstance(gen, gens.CLMGenerator)
    for _ in range(200):
        value = gen.next()
        assert isinstance(value, float)
        assert math.isfinite(value)
