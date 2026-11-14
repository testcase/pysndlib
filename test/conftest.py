"""Shared fixtures for the pysndlib test suite."""
import pytest

import pysndlib.clm as clm

SRATE = 44100


@pytest.fixture
def srate():
    """The sample rate all test renders use."""
    return SRATE


@pytest.fixture
def sine_wav(tmp_path):
    """Render a known mono sine (440 Hz, 1000 samples) and return
    ``(path, srate, nsamples, frequency)``."""
    path = str(tmp_path / "sine.wav")
    with clm.Sound(path, channels=1, srate=SRATE) as snd:
        osc = clm.make_oscil(440.0)
        for i in range(1000):
            clm.outa(i, clm.oscil(osc))
    return path, SRATE, 1000, 440.0


@pytest.fixture(autouse=True)
def clean_clm_default():
    """Snapshot and restore ``clm.default`` around every test so no test can
    leak output/reverb/srate state into another, and fail if a test leaves a
    pending sndlib error in the handler slot."""
    try:
        clm.check_sndlib_errors()
    except clm.SNDLibError:
        pass  # swallow anything left over from a previous (failed) test
    saved = vars(clm.default).copy()
    yield
    vars(clm.default).clear()
    vars(clm.default).update(saved)
    clm.check_sndlib_errors()  # raises (-> teardown error) if the test left an error pending
