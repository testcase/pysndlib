"""Tests for pysndlib.env — envelope utilities."""
import numpy as np
import pytest

import pysndlib.env as env



def test_interleave_lists():
    assert env.interleave([1, 2], [3, 4]) == [1, 3, 2, 4]


def test_interleave_ndarrays():
    out = env.interleave(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out, [1.0, 3.0, 2.0, 4.0])


def test_interleave_rejects_other_types():
    with pytest.raises(TypeError):
        env.interleave("nope", "nope")


def test_scale_and_offset_list():
    assert env.scale_and_offset([1.0, 2.0], 2.0, 1.0) == [3.0, 5.0]


def test_scale_and_offset_ndarray():
    out = env.scale_and_offset(np.array([1.0, 2.0]), 0.5, 1.0)
    np.testing.assert_allclose(out, [1.5, 2.0])


def test_scale_and_offset_rejects_other_types():
    # regression (0.8.0): silently returned None
    with pytest.raises(TypeError):
        env.scale_and_offset("nope", 2.0)


def test_multiply_and_add_envelopes():
    prod = env.multiply_envelopes([0, 1, 100, 2], [0, 2, 100, 3])
    total = env.add_envelopes([0, 1, 100, 2], [0, 2, 100, 3])
    assert len(prod) == len(total) == 4


def test_max_envelope_returns_max_y():
    # max_envelope returns the maximum y VALUE (not a new envelope)
    assert env.max_envelope([0, 0, 50, 2, 100, 1]) == 2


def test_min_envelope_returns_min_y():
    assert env.min_envelope([0, 2, 50, 1, 100, 0]) == 0


def test_integrate_envelope_known_area():
    # triangle from (0,0) to (1,1): area 0.5
    assert env.integrate_envelope([0, 0, 1, 1]) == pytest.approx(0.5)
    # rectangle 1 unit high across 100 units: area 100
    assert env.integrate_envelope([0, 1, 100, 1]) == pytest.approx(100.0)


def test_envelope_last_x():
    assert env.envelope_last_x([0, 0, 25, 1, 100, 0]) == 100


def test_stretch_envelope_known_result():
    # verified against the drone instrument's envelope in 0.8.0 testing
    result = env.stretch_envelope([0, 0, 100, 1], 25, 10, 75, 90)
    assert result == [0, 0, 10, 0.25, 90, 0.75, 100.0, 1]


def test_scale_envelope():
    out = env.scale_envelope([0, 0, 100, 1], 2.0)
    assert out[1] == 0 and out[3] == 2.0


def test_reverse_envelope():
    out = env.reverse_envelope([0, 0, 50, 1, 100, 2])
    xs = out[::2]
    ys = out[1::2]
    assert xs == [0, 50, 100]
    assert ys == [2, 1, 0]


def test_concatenate_envelopes():
    out = env.concatenate_envelopes([0, 0, 100, 1], [0, 0, 100, 1])
    assert len(out) >= 8
    assert out[0] == 0


def test_normalize_envelope():
    out = env.normalize_envelope([0, 0, 50, 5, 100, 0], new_max=1.0)
    assert max(out[1::2]) == pytest.approx(1.0)
