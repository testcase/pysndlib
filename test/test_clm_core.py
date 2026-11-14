"""Tests for pysndlib.clm core: Sound context manager, out/in, file I/O,
and unit-conversion utilities."""
import numpy as np
import pytest

import pysndlib.clm as clm
import pysndlib.sndlib as sndlib



# ---- Sound context manager ----

def test_sound_renders_to_file(sine_wav, tmp_path):
    path, sr, nsamples, freq = sine_wav
    assert clm.get_length(path) == nsamples
    assert clm.get_srate(path) == sr
    assert clm.get_channels(path) == 1
    arr, _ = clm.file2ndarray(path)
    assert arr.shape == (1, nsamples)
    assert float(np.abs(arr).max()) == pytest.approx(1.0, abs=1e-3)


def test_sound_ndarray_output():
    out = np.zeros((2, 500))
    with clm.Sound(out, srate=44100):
        o1, o2 = clm.make_oscil(440.0), clm.make_oscil(660.0)
        for i in range(500):
            clm.outa(i, clm.oscil(o1))
            clm.outb(i, clm.oscil(o2))
    assert float(np.abs(out[0]).max()) > 0.5
    assert float(np.abs(out[1]).max()) > 0.5


def test_sound_restores_default_state():
    prev_output, prev_reverb = clm.default.output, clm.default.reverb
    out = np.zeros((1, 100))
    with clm.Sound(out, srate=44100, reverb=lambda: (lambda: None), reverb_channels=1):
        assert clm.default.output is out
        osc = clm.make_oscil(220.0)
        for i in range(100):
            clm.outa(i, clm.oscil(osc))
    assert clm.default.output is prev_output
    assert clm.default.reverb is prev_reverb


def test_sound_statistics_prints(sine_wav, tmp_path, capsys):
    path = str(tmp_path / "stats.wav")
    with clm.Sound(path, channels=2, srate=44100, statistics=True):
        o1, o2 = clm.make_oscil(440.0), clm.make_oscil(660.0)
        for i in range(500):
            clm.outa(i, clm.oscil(o1))
            clm.outb(i, clm.oscil(o2))
    captured = capsys.readouterr()
    assert "channels: 2" in captured.out
    assert "maxamp" in captured.out


def test_sound_continue_old_file(tmp_path):
    """Regression (0.8.0): __enter__ used a never-set self.filename."""
    path = str(tmp_path / "continue.wav")
    with clm.Sound(path, channels=1, srate=44100):
        osc = clm.make_oscil(440.0)
        for i in range(100):
            clm.outa(i, clm.oscil(osc))
    with clm.Sound(path, channels=1, srate=44100, continue_old_file=True):
        osc = clm.make_oscil(440.0)
        for i in range(100, 200):
            clm.outa(i, clm.oscil(osc))
    assert clm.get_length(path) == 200


def test_sound_scaled_to(tmp_path):
    path = str(tmp_path / "scaled.wav")
    with clm.Sound(path, channels=1, srate=44100, scaled_to=0.25):
        osc = clm.make_oscil(440.0)
        for i in range(500):
            clm.outa(i, clm.oscil(osc))
    arr, _ = clm.file2ndarray(path)
    assert float(np.abs(arr).max()) == pytest.approx(0.25, abs=0.02)


# ---- out/in ----

def test_out_any_writes_channel():
    out = np.zeros((2, 4))
    clm.out_any(1, 0.5, 0, out)
    clm.out_any(2, -0.25, 1, out)
    assert out[0][1] == 0.5
    assert out[1][2] == -0.25


def test_out_channels_are_independent():
    out = np.zeros((2, 10))
    for i in range(10):
        clm.outa(i, 0.5, out)
        clm.outb(i, -0.5, out)
    assert np.all(out[0] == 0.5)
    assert np.all(out[1] == -0.5)


def test_in_any_reads_ndarray():
    inp = np.arange(20, dtype=np.double).reshape((2, 10))
    assert clm.in_any(3, 0, inp) == 3.0
    assert clm.in_any(3, 1, inp) == 13.0


def test_in_any_reads_list():
    inp = [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]
    assert clm.in_any(2, 0, inp) == 2.0
    assert clm.in_any(2, 1, inp) == 12.0


def test_in_any_reads_generator_function():
    def source(loc, chan):
        return 7.0
    assert clm.in_any(0, 0, source) == 7.0


# ---- file <-> ndarray ----

def test_ndarray2file_file2ndarray_round_trip(tmp_path):
    path = str(tmp_path / "roundtrip.wav")
    data = np.random.default_rng(42).uniform(-0.5, 0.5, (2, 300))
    clm.ndarray2file(path, data, sr=22050)
    back, sr = clm.file2ndarray(path)
    assert sr == 22050
    np.testing.assert_allclose(back, data, atol=1e-6)


def test_file2ndarray_channel_selection(sine_wav):
    path, sr, nsamples, freq = sine_wav
    all_chans, _ = clm.file2ndarray(path)
    one_chan, _ = clm.file2ndarray(path, channel=0)
    assert all_chans.shape == (1, nsamples)
    assert one_chan.shape == (1, nsamples)
    np.testing.assert_allclose(all_chans, one_chan)


def test_file2ndarray_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        clm.file2ndarray(str(tmp_path / "missing.wav"))


# ---- unit conversions and small utilities ----

@pytest.mark.parametrize("hz", [1.0, 440.0, 22050.0])
def test_hz2radians_radians2hz_round_trip(hz, srate):
    rads = clm.hz2radians(hz)
    assert clm.radians2hz(rads) == pytest.approx(hz, rel=1e-9)


def test_seconds2samples_round_trip(srate):
    samples = clm.seconds2samples(1.5)
    assert samples == int(1.5 * srate)


def test_is_odd_is_even():
    assert clm.is_odd(3) and not clm.is_odd(4)
    assert clm.is_even(4) and not clm.is_even(3)


def test_mus_fft_finds_sine_bin(srate):
    n = 512
    freq = 1000.0
    x = np.arange(n)
    rl = np.sin(2 * np.pi * freq * x / srate)
    im = np.zeros(n)
    clm.mus_fft(rl, im, n, 1)
    mags = np.abs(rl) + np.abs(im)
    peak_bin = int(np.argmax(mags[: n // 2]))
    assert abs(peak_bin - freq * n / srate) <= 1


def test_polynomial_constant_term_first():
    coeffs = np.array([1.0, 2.0, 3.0, 4.0])
    # coeffs[0] is the constant term (mus_polynomial convention)
    assert clm.polynomial(coeffs, 0.0) == pytest.approx(1.0)


# ---- play path ----

def test_play_raises_without_player(tmp_path):
    path = str(tmp_path / "play.wav")
    old_player = clm.default.player
    try:
        clm.default.player = None
        with pytest.raises(RuntimeError, match="no player"):
            with clm.Sound(path, channels=1, srate=44100, play=True):
                clm.outa(0, 0.5)
    finally:
        clm.default.player = old_player


def test_play_list_form_substitutes_placeholder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(clm.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    path = str(tmp_path / "play.wav")
    clm.default.player = ['echo', 'playing {file}']  # restored by clean_clm_default
    with clm.Sound(path, channels=1, srate=44100, play=True):
        clm.outa(0, 0.5)
    assert calls and calls[0] == ['echo', f'playing {path}']


def test_play_list_form_appends_file_without_placeholder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(clm.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    path = str(tmp_path / "play2.wav")
    clm.default.player = ['some-player', '-flag']
    with clm.Sound(path, channels=1, srate=44100, play=True):
        clm.outa(0, 0.5)
    assert calls and calls[0] == ['some-player', '-flag', path]


def test_play_string_form_unchanged(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(clm.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    path = str(tmp_path / "play3.wav")
    clm.default.player = 'my-player'
    with clm.Sound(path, channels=1, srate=44100, play=True):
        clm.outa(0, 0.5)
    assert calls and calls[0] == ['my-player', path]


def test_play_escapes_single_quotes_in_placeholder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(clm.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    path = str(tmp_path / "it's a file.wav")
    clm.default.player = ['echo', 'path={file}']
    with clm.Sound(path, channels=1, srate=44100, play=True):
        clm.outa(0, 0.5)
    escaped = path.replace("'", "''")   # PowerShell single-quote escaping
    assert calls and calls[0] == ['echo', f'path={escaped}']
