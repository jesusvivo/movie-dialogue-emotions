"""Unit tests for timeline bucketing."""
from __future__ import annotations

from src.analysis import split_into_thirds


def test_split_into_thirds_evenly():
    out = split_into_thirds(list(range(9)))
    assert out == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_split_into_thirds_uneven_covers_all_items():
    n = 10
    out = split_into_thirds(list(range(n)))
    flat = [x for bucket in out for x in bucket]
    assert flat == list(range(n))
    # No bucket should be more than 1 off from the smallest.
    lengths = [len(b) for b in out]
    assert max(lengths) - min(lengths) <= 1


def test_split_into_thirds_small_input():
    out = split_into_thirds([0, 1])
    flat = [x for bucket in out for x in bucket]
    assert flat == [0, 1]
    assert len(out) == 3
