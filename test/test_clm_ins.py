"""Tests for the instruments in pysndlib.clm_ins (vox, drone, canter)."""
import numpy as np
import pytest

import pysndlib.clm as clm
from pysndlib.clm_ins import canter, drone, vox


def test_instruments_package_imports():
    """The compiled instruments package must import cleanly; this catches
    Cython signature drift between instruments/*.py and the clm module
    (found by the notebook probes after the LLP64 signature change)."""
    import pysndlib.instruments  # noqa: F401
    from pysndlib.instruments import fm_violin, gong, pluck  # noqa: F401
    assert callable(fm_violin) and callable(gong) and callable(pluck)



def _assert_nonsilent(path):
    arr, _ = clm.file2ndarray(path)
    assert float(np.abs(arr).max()) > 0.0, f"{path} rendered silence"


def test_vox(tmp_path):
    path = str(tmp_path / "vox.wav")
    with clm.Sound(path, channels=1, srate=44100):
        vox(0, 0.05, 170, .4,
            [0, 0, 25, 1, 75, 1, 100, 0],          # ampfun
            [0, 0, 5, .5, 10, 0, 100, 1],          # freqfun
            .1,                                     # freqscl
            [0, 'E', 25, 'AE', 35, 'ER', 65, 'ER', 75, 'I', 100, 'UH'],
            [.8, .15, .05],                         # formant amps
            [.005, .0125, .025],                    # formant indices
            .05, .1)                                # vibscl, deg
    _assert_nonsilent(path)


def test_drone(tmp_path):
    path = str(tmp_path / "drone.wav")
    with clm.Sound(path, channels=1, srate=44100):
        drone(0, 0.1, 220.0, 0.5,
              [0, 0, 100, 1],        # ampfun
              [1, 1],                # synth: partials pairs (harmonic, amp)
              0.01, 0.01,            # ampat, ampdc
              0,                     # amtrev
              0, 1.0,                # deg, dis
              .01, 6)                # rvibamt, rvibfreq
    _assert_nonsilent(path)


def test_canter(tmp_path):
    path = str(tmp_path / "canter.wav")
    env1 = [0, 0, 100, 1]
    fmt = [0, 200, 50, 400, 100, 600]
    # canter's k = 100*log(pitch/400)/log(910/400): pitch must lie within
    # the 400-910 Hz range for the envelope lookups to be meaningful
    with clm.Sound(path, channels=1, srate=44100):
        canter(0, 0.05, 600.0, 0.5,
               0, 1.0, 0,            # deg, dis, pcrev
               env1, env1, env1,     # ampfun, ranfun, skewfun
               1.0, .1, 6,           # skewpc, ranpc, ranfreq
               env1,                 # indexfun
               .01, .01,             # atdr, dcdr
               env1, env1, fmt,      # harmonic 1: amp, ind, fmt
               env1, env1, fmt,      # harmonic 2
               env1, env1, fmt,      # harmonic 3
               env1, env1, fmt)      # harmonic 4
    _assert_nonsilent(path)
