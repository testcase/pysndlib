"""Parametrized sweep over the mus_any generators of pysndlib.clm.

Adding a generator to the sweep is one registry line: the ``is_*`` checker
and the procedural runner are derived from the factory name.
"""
import math

import numpy as np
import pytest

import pysndlib.clm as clm

SINE_WAVE = None  # filled lazily by _sine_wave()


def _sine_wave():
    global SINE_WAVE
    if SINE_WAVE is None:
        SINE_WAVE = clm.partials2wave([1, 1])
    return SINE_WAVE


# (factory name, args, kwargs)
CLM_GENERATORS = [
    # sinusoids and standard waveforms
    ("oscil", (440.0,), {}),
    ("env", ([0, 0, 100, 1],), {"duration": 0.01}),
    ("pulsed_env", ([0, 0, 100, 1], 0.01, 440.0), {}),
    ("table_lookup", (440.0,), {"wave": _sine_wave()}),
    ("table_lookup_with_env", (440.0, [0, 1, 50, 1, 100, 0]), {}),
    ("polywave", (440.0, [0.0, 1.0]), {}),
    ("polyshape", (440.0,), {}),
    ("triangle_wave", (440.0,), {}),
    ("square_wave", (440.0,), {}),
    ("sawtooth_wave", (440.0,), {}),
    ("pulse_train", (440.0,), {}),
    ("wave_train", (440.0,), {"wave": _sine_wave()}),
    ("wave_train_with_env", (440.0, [0, 1, 50, 1, 100, 0]), {}),
    ("ssb_am", (440.0,), {}),
    ("asymmetric_fm", (440.0,), {}),
    # n-sum generators
    ("ncos", (440.0, 3), {}),
    ("nsin", (440.0, 3), {}),
    ("nrxysin", (440.0,), {}),
    ("nrxycos", (440.0,), {}),
    ("rxykcos", (440.0,), {}),
    ("rxyksin", (440.0, 0.0), {}),
    # noise
    ("rand", (440.0,), {}),
    ("rand_interp", (440.0,), {}),
    # filters
    ("one_pole", (0.5, 0.5), {}),
    ("one_zero", (0.5, 0.5), {}),
    ("two_pole", (), {"a0": 1.0, "b1": 0.5, "b2": 0.25}),
    ("two_zero", (), {"a0": 1.0, "a1": 0.5, "a2": 0.25}),
    ("formant", (440.0, 0.9), {}),
    ("firmant", (440.0, 0.9), {}),
    ("filter", (3, [0.5, 0.5, 0.5], [0.0, 0.25, 0.25]), {}),
    ("fir_filter", (3, [0.33, 0.33, 0.33]), {}),
    ("iir_filter", (2, [1.0, -0.5]), {}),
    ("dcblock", (), {}),
    # biquads
    ("biquad", (1.0, 0.0, 0.0, 0.0, 0.0), {}),
    ("biquad_notch", (440.0, 0.9), {}),
    ("biquad_low_pass", (440.0, 1.0), {}),
    ("biquad_high_pass", (440.0, 1.0), {}),
    ("biquad_band_pass", (440.0, 1.0), {}),
    ("biquad_band_reject", (440.0, 1.0), {}),
    ("biquad_all_pass", (440.0, 1.0), {}),
    # delay lines
    ("delay", (100,), {}),
    ("comb", (0.5, 100), {}),
    ("notch", (0.5, 100), {}),
    ("all_pass", (0.5, 0.5, 100), {}),
    ("one_pole_all_pass", (100, 0.5), {}),
    ("filtered_comb", (0.5, 100), {"filter": None}),
    ("moving_average", (10,), {}),
    ("moving_max", (10,), {}),
    ("moving_norm", (10,), {}),
]

# generators whose two instances produce different values (private RNG state)
NON_DETERMINISTIC = {"rand", "rand_interp"}

IDS = [entry[0] for entry in CLM_GENERATORS]


def make(name):
    """Build a generator from a registry name."""
    factory_name, args, kwargs = next(e for e in CLM_GENERATORS if e[0] == name)
    factory = getattr(clm, "make_" + factory_name)
    if factory_name == "filtered_comb":
        kwargs = {"filter": clm.make_one_pole(0.5, 0.5)}
    return factory(*args, **kwargs)


@pytest.mark.parametrize("name", IDS)
def test_generator_constructs_as_mus_any(name):
    gen = make(name)
    assert isinstance(gen, clm.mus_any)
    checker = getattr(clm, "is_" + name, None)
    if checker is not None:
        assert checker(gen), f"is_{name} should accept its own generator"
        other = clm.make_oscil(100.0)
        if name != "oscil":
            assert not checker(other), f"is_{name} should reject an oscil"


@pytest.mark.parametrize("name", IDS)
def test_generator_output_is_finite(name):
    gen = make(name)
    for _ in range(500):
        value = gen()
        assert isinstance(value, float)
        assert math.isfinite(value), f"{name} produced {value}"


@pytest.mark.parametrize("name", IDS)
def test_procedural_call_matches_callable(name):
    """``gen(arg)`` must equal the procedural ``clm.<name>(gen, ...)`` for two
    identically constructed generators."""
    if name in NON_DETERMINISTIC:
        pytest.skip("private RNG state: instances differ")
    runner = getattr(clm, name, None)
    if runner is None:
        pytest.skip(f"no procedural runner clm.{name}")
    g1, g2 = make(name), make(name)
    try:
        expected = runner(g2, 0.25)
    except TypeError:
        # some runners (e.g. clm.env) take no extra argument; the argument
        # is ignored by the generator, so the first outputs still match
        expected = runner(g2)
    assert g1(0.25) == expected


@pytest.mark.parametrize("name", IDS)
def test_generator_describe_and_str(name):
    gen = make(name)
    assert "mus_any" in str(gen)


# ---- waveform spot checks (known-value behavior) ----

def test_oscil_zero_crossings_match_frequency(srate):
    freq = 440.0
    osc = clm.make_oscil(freq)
    samples = np.array([clm.oscil(osc) for _ in range(2000)])
    crossings = np.sum(np.diff(np.signbit(samples)))
    cycles = crossings / 2.0
    expected = 2000 * freq / srate  # ~19.9 cycles
    assert abs(cycles - expected) < 1.0


def test_oscil_amplitude_is_bounded_by_one():
    osc = clm.make_oscil(440.0)
    samples = [clm.oscil(osc) for _ in range(2000)]
    assert max(samples) <= 1.0 + 1e-9
    assert min(samples) >= -1.0 - 1e-9


def test_env_reaches_endpoints():
    env = clm.make_env([0, 0, 100, 1], duration=0.01)
    values = [clm.env(env) for _ in range(441)]
    assert values[0] == pytest.approx(0.0, abs=1e-3)
    assert values[-1] == pytest.approx(1.0, abs=1e-3)
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in values)


def test_rand_stays_within_amplitude():
    rand = clm.make_rand(440.0, 0.5)
    values = [clm.rand(rand) for _ in range(1000)]
    assert max(values) <= 0.5 + 1e-9
    assert min(values) >= -0.5 - 1e-9


def test_triangle_wave_is_bounded():
    gen = clm.make_triangle_wave(440.0)
    values = [clm.triangle_wave(gen) for _ in range(1000)]
    assert max(values) <= 1.0 + 1e-9
    assert min(values) >= -1.0 - 1e-9


def test_delay_reproduces_input_after_size_samples():
    delay = clm.make_delay(100)
    outputs = [clm.delay(delay, 1.0 if i == 0 else 0.0) for i in range(102)]
    # the impulse written at index 0 reappears at index 100
    assert outputs[99] == pytest.approx(0.0, abs=1e-12)
    assert outputs[100] == pytest.approx(1.0, abs=1e-12)
    assert outputs[101] == pytest.approx(0.0, abs=1e-12)


def test_one_pole_tracks_constant_input():
    # sndlib's one-pole convention: y(n) = a0*x(n) - b1*y(n-1), so the DC
    # gain of make_one_pole(a0, b1) is a0 / (1 + b1); (0.5, -0.5) gives 1.0
    gen = clm.make_one_pole(0.5, -0.5)
    for _ in range(200):
        clm.one_pole(gen, 1.0)
    assert clm.one_pole(gen, 1.0) == pytest.approx(1.0, abs=1e-9)


def test_fir_filter_passes_dc():
    coeffs = [1.0 / 3.0] * 3
    gen = clm.make_fir_filter(3, coeffs)
    for _ in range(10):
        clm.fir_filter(gen, 0.5)
    assert clm.fir_filter(gen, 0.5) == pytest.approx(0.5, abs=1e-12)


def test_moving_average_smooths_impulse():
    gen = clm.make_moving_average(4)
    gen(1.0)
    outputs = [gen(0.0) for _ in range(4)]
    assert outputs[0] == pytest.approx(0.25, abs=1e-9)
    assert outputs[3] == pytest.approx(0.0, abs=1e-9)


# ---- banks ----

def test_oscil_bank_sums_oscils():
    # oscil_bank takes frequencies in radians (per the make_oscil_bank docstring)
    freqs = [clm.hz2radians(440.0), clm.hz2radians(660.0)]
    phases = [0.0, 0.0]
    bank = clm.make_oscil_bank(freqs, phases)
    assert clm.is_oscil_bank(bank)
    single = clm.make_oscil(440.0)
    # with default amplitudes the bank equals the sum of its members
    o1, o2 = clm.make_oscil(440.0), clm.make_oscil(660.0)
    for _ in range(10):
        expected = clm.oscil(o1) + clm.oscil(o2)
        assert clm.oscil_bank(bank) == pytest.approx(expected, abs=1e-6)


def test_comb_bank_and_all_pass_bank_construct():
    combs = [clm.make_comb(0.5, 100 + i) for i in range(3)]
    bank = clm.make_comb_bank(combs)
    assert isinstance(bank, clm.mus_any)
    aps = [clm.make_all_pass(0.5, 0.5, 100 + i) for i in range(3)]
    abank = clm.make_all_pass_bank(aps)
    assert isinstance(abank, clm.mus_any)


def test_formant_bank_constructs():
    filters = [clm.make_formant(440.0 + i * 110.0, 0.9) for i in range(3)]
    bank = clm.make_formant_bank(filters, [0.5, 0.3, 0.2])
    assert isinstance(bank, clm.mus_any)
    for _ in range(100):
        assert math.isfinite(bank(0.0))


def test_locsig_places_in_channels(sine_wav):
    path, sr, n, freq = sine_wav
    with clm.Sound(path, channels=2, srate=sr):
        loc = clm.make_locsig(0.0, 1.0, 0.0)
        clm.locsig(loc, 0, 0.5)
    arr, _ = clm.file2ndarray(path)
    assert abs(arr[0][0]) > 0.4
    assert abs(arr[1][0]) < 1e-6
