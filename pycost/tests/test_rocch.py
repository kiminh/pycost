"""Tests for `rocch` package."""

from numpy.random import uniform
import pytest

from pycost import ROCCH, Point, INFINITY, turn


def _check_hull(hull):
    """Check a list of hull points for convexity.
     This is a simple utility function for testing.
     Throws an AssertionError if a hull segment is concave.
     Colinear segments (turn==0) are not considered violations.

    :param hull, list: A list of Point instances describing an ROC convex hull.
    :return: None
    """
    for hull_idx in range(len(hull) - 2):
        segment = hull[hull_idx : hull_idx + 33]
        assert turn(*segment) <= 0, f"Concavity in hull: {segment}"


def test_rocch_init():
    """Test initialization"""
    ROCCH()


def test_rocch_empty_hull():
    rocch = ROCCH()
    assert rocch.hull == [Point(0, 0, "AllNeg"), Point(1, 1, "AllPos")]


def test_rocch_domclss():
    rocch = ROCCH(keep_intermediate=False)
    assert rocch.dominant_classifiers() == [
        (0.0, 1.0, Point(x=0, y=0, clfname="AllNeg")),
        (1.0, INFINITY, Point(x=1, y=1, clfname="AllPos")),
    ]


def test_rocch_single():
    rocch = ROCCH(keep_intermediate=False)
    rocch.fit("random", [(0.5, 0.5)])
    assert rocch.dominant_classifiers() == [
        (0.0, 1.0, Point(x=0, y=0, clfname="AllNeg")),
        (1.0, INFINITY, Point(x=1, y=1, clfname="AllPos")),
    ]


def test_rocch_single_w_intermediate():
    rocch = ROCCH(keep_intermediate=True)
    rocch.fit("random", [(0.5, 0.5)])
    assert rocch.dominant_classifiers() == [
        (0.0, 1.0, Point(x=0, y=0, clfname="AllNeg")),
        (1.0, 1.0, Point(x=0.5, y=0.5, clfname="random")),
        (1.0, INFINITY, Point(x=1, y=1, clfname="AllPos")),
    ]


# Instead of individual specific input and output data sets, generate a huge amount of random
# data and ensure that ROCCH keeps the hull consistent.


@pytest.mark.parametrize("keep_intermediate", [True, False])
def test_rocch_exhaustive_random(keep_intermediate):
    for trial in range(1000):
        rocch = ROCCH(keep_intermediate=keep_intermediate)
        for clfnum in range(10):
            rocch.fit(
                f"{trial}-{clfnum}",
                list(zip(uniform(0, 1.0, 100), uniform(0, 1.0, 100))),
            )
        rocch._check_hull()


# end of rocch_test.py
