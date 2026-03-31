"""Tests for preprocessing helpers."""

import numpy as np

from utils.preprocessing import build_uniform_sample_indices


def test_build_uniform_sample_indices_empty_cases():
    assert build_uniform_sample_indices(0, 10).size == 0
    assert build_uniform_sample_indices(10, 0).size == 0


def test_build_uniform_sample_indices_span_and_count():
    indices = build_uniform_sample_indices(total_frames=100, max_frames=7)
    assert indices.dtype == np.int32
    assert len(indices) == 7
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 99
    assert np.all(np.diff(indices) > 0)


def test_build_uniform_sample_indices_when_frames_are_fewer_than_budget():
    indices = build_uniform_sample_indices(total_frames=5, max_frames=20)
    assert np.array_equal(indices, np.asarray([0, 1, 2, 3, 4], dtype=np.int32))
