"""

Metrics to calculate and manipulate the ROC Convex Hull on a classification task given scores.

"""

# Author: Tom Fawcett <tom.fawcett@gmail.com>
from collections import namedtuple
from math import sqrt
from typing import List, Dict, Tuple


# DESCRIPTION:
#
# This program computes the convex hull of a set of ROC points
# (technically, the upper left triangular convex hull, bounded
# by (0,0) and (1,1)).  The ROC Convex Hull is used to find dominant
# (and locally best) classifiers in ROC space.  For more information
# on the ROC convex hull and its uses, see the references below.
#
# FP and TP are the False Positive (X axis) and True Positive (Y axis)
# values for the point.
#
#
# REFERENCES:
##
# The first paper below is probably best for an introduction and
# general discussion of the ROC Convex Hull and its uses.
##
# 1) Provost, F. and Fawcett, T. "Analysis and visualization of
# classifier performance: Comparison under imprecise class and cost
# distributions".  In Proceedings of the Third International
# Conference on Knowledge Discovery and Data Mining (KDD-97),
# pp.43-48. AAAI Press.  Available from:
# http://www.croftj.net/~fawcett/papers/KDD-97.ps.gz
##
# 2) Provost, F. and Fawcett, T. "Robust Classification Systems for
# Imprecise Environments".  To be presented at AAAI-98.
# Submitted paper available from:
# http://www.croftj.net/~fawcett/papers/aaai98.ps.gz
##
# 3) Provost, F., Fawcett, T., and Kohavi, R.  "Building the Case
# Against Accuracy Estimation for Comparing Induction Algorithms".
# Available from:  TODO: FIX THESE URLSs
# http://www.croftj.net/~fawcett/papers/ICML98-submitted.ps.gz
##
##
# BUG REPORTS / SUGGESTIONS / QUESTIONS: Tom Fawcett <tom.fawcett@gmail.com>
##
##

"""

Typical use is:
 
 rocch = ROCCH(keep_intermediate=False)
 for clf in classifiers:
     y_scores = clf.decision_function(y_test)
     rocch.fit(clfname, roc_curve(y_scores, y_true))
 ...
 plt.plot(rocch.hull())
 rocch.describe()
 
"""

Point: namedtuple = namedtuple( "Point", ["x", "y", "clfname"] )
Point.__new__.__defaults__ = ("",)  # make clfname optional

INFINITY: float = float( "inf" )


class ROCCH( object ):
    """ROC Convex Hull.

    Some other stuff.
    """
    _hull: List[Point]

    def __init__(self, keep_intermediate=False):
        """Initialize the object."""
        self.keep_intermediate = keep_intermediate
        self.classifiers: Dict[str, List[Tuple]] = { }
        self._hull = [Point( 0, 0, "AllNeg" ), Point( 1, 1, "AllPos" )]

    def fit(self, clfname: str, points):
        """Fit (add) a classifier's ROC points to the ROCCH.

        :param clfname: A classifier name or identifier.  This is only used to record the
        identity of the classifier producing the points. It can be anything, such as a
        (classifier, threshold) pair.
        TODO: Let clfname be a string or a list; add some way to incorporate info per point so we
        can associate each point with a parameter.

        :param points: A sequence of ROC points, contained in a list or array.  Each point should
        be an (FP, TP) pair.  TODO: Make this more general.

        :return: None
        """
        points_instances = [Point( x, y, clfname ) for (x, y) in points]
        points_instances.extend( self._hull )
        points_instances.sort( key=lambda pt: pt.x )
        hull = []

        # TODO: Make this more efficient by simply using pointers rather than append-pop.

        while points_instances:
            hull.append( points_instances.pop( 0 ) )
            # Now test the top three on new_hull
            test_top = True
            while len( hull ) >= 3 and test_top:
                turn_dir = turn( *hull[-3:] )
                if turn_dir > 0:  # CCW turn, this introduced a concavity.
                    hull.pop( -2 )
                elif turn_dir == 0:  # Co-linear, should we keep it?
                    if not self.keep_intermediate:
                        # No, treat it as if it's under the hull
                        hull.pop( -2 )
                    else:  # Treat this as convex
                        test_top = False
                else:  # CW turn, this is convex
                    test_top = False
        self._hull = hull

    def _check_hull(self) -> None:
        """Check a list of hull points for convexity.
           This is a simple utility function for testing.
           Throws an AssertionError if a hull segment is concave or if the terminal AllNeg and
           AllPos are not present.
           Colinear segments (turn==0) will be considered violations unless keep_intermediate is on.

        """
        hull = self._hull
        assert len( hull ) >= 2, "Hull is damaged"
        assert hull[0].clfname == "AllNeg", "First hull point is not AllNeg"
        assert hull[-1].clfname == "AllPos", "Last hull point is not AllPos"
        for hull_idx in range( len( hull ) - 2 ):
            segment = hull[hull_idx:hull_idx + 3]
            turn_val = turn( *segment )
            assert turn_val <= 0, f"Concavity in hull: {segment}"
            if not self.keep_intermediate:
                assert turn_val < 0, "Intermediate (colinear) point in hull"

    @property
    def hull(self) -> List[Tuple]:
        """
        Return a list of points constituting the convex hull of classifiers in ROC space.
        Returns a list of tuples (FP, TP, CLF) where each (FP,TP) is a point in ROC space
        and CLF is the classifier producing that performance point.
        """
        # Defined just in case postprocessing needs to be done.
        return self._hull

    def dominant_classifiers(self) -> List[Tuple]:
        """
        Return a list describing the hull in terms of the dominant classifiers.

        Start at point (1,1) and work counter-clockwise down the hull to (0,0).
        Iso-performance line slope starts at 0.0 and works up to infinity.

        :return: A list consisting of (prob_min, prob_max, point) where
        :rtype: List[Tuple]
        """
        slope = 0.0
        last_point = None
        last_slope = None
        segment_right_boundary: Point = None
        dominant_list: List[Tuple] = []
        # TODO: Check for hull uninitialized.
        point: Point
        for point in self._hull:
            if last_point is not None:
                slope: float = calculate_slope( point, last_point )
            else:
                segment_right_boundary = point
            if last_slope is not None:
                if self.keep_intermediate or last_slope != slope:
                    dominant_list.append( (last_slope, slope, segment_right_boundary) )
                last_slope = slope
                segment_right_boundary = point
            else:  # last_slope is undefined
                last_slope = slope
            last_point = point
        if last_slope != INFINITY:
            slope = INFINITY
        # Output final point
        dominant_list.append( (last_slope, slope, segment_right_boundary) )
        return dominant_list

    def best_classifiers_for_conditions(self, class_ratio=1.0, cost_ratio=1.0):
        """
        Given a set of operating conditions (class and cost ratios), return best classifiers.

        Given a class ratio (P/N) and a cost ratio (cost(FP),cost(FN)), return a set of
        classifiers that will perform optimally for those conditions.  The class ratio is the
        fraction
        of positives per negative.  The cost ratio is the cost of a False Positive divided by the
        cost
        of a False Negative.

        The return value will be a list of either one or two classifiers.  If the conditions
        identify a single best classifier, the result will be simply:
        [ (clf, 1.0) ]
        indicating that clf should be chosen.

        If the conditions are between the performance of two classifiers, the result will be:
        [ (clf1, p1), (clf2, p2) ]
        indicating that clf1's decisions should be sampled at a rate of p1 and clf2's at a rate
        of p2, with p1 and p2 summing to 1.


        :param class_ratio:
        :type class_ratio:
        :param cost_ratio:
        :type cost_ratio:
        :return:
        :rtype:
        """
        assert 0 < class_ratio < 1.0, "Class ratio must be between 0 and 1"
        assert 0 < cost_ratio < 1.0, "Cost ratio must be between 0 and 1"


def calculate_slope(pt1, pt2: Point):
    """
    Return the slope from pt1 to pt2, or inf if slope is infinite
    :param pt1:
    :type pt1: Point
    :param pt2:
    :type pt2: Point
    :return:
    :rtype: float
    """
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    if dx == 0:
        return INFINITY
    else:
        return dy / dx


def _check_hull(hull):
    """Check a list of hull points for convexity.
     This is a simple utility function for testing.
     Throws an AssertionError if a hull segment is concave.
     Colinear segments (turn==0) are not considered violations.

    :param hull: A list of Point instances describing an ROC convex hull.
    :return: None 
    """
    for hull_idx in range( len( hull ) - 2 ):
        segment = hull[hull_idx:hull_idx + 3]
        assert turn( *segment ) <= 0, f"Concavity in hull: {segment}"


def ROC_order(pt1, pt2: Point) -> bool:
    """Predicate for determining ROC_order for sorting.

    Either pt1's x is ahead of pt2's x, or the x's are equal and pt1's y is ahead of pt2's y.
    """
    return (pt1.x < pt2.x) or (pt1.x == pt2.x and pt1.y < pt2.y)


def compute_theta(p1, p2: Point) -> float:
    """Compute theta, an ordering function on a point pair.

    Theta has the same properties as the angle between the horizontal axis and
    the line segment between the points, but is much faster to compute than
    arctangent.  Range is 0 to 360.  Defined on P.353 of _Algorithms in C_.

    """
    dx = p2.x - p1.x
    ax = abs( dx )
    dy = p2.y - p1.y
    ay = abs( dy )
    if dx == 0 and dy == 0:
        t = 0
    else:
        t = dy / (ax + ay)
    # Adjust for quadrants two through four
    if dx < 0:
        t = 2 - t
    elif dy < 0:
        t = 4 + t
    return t * 90.0


def euclidean(p1, p2: Point) -> float:
    """Compute Euclidean distance.
    """
    return sqrt( (p1.x - p2.x)**2 + (p1.y - p2.y)**2 )


def turn(a, b, c: Point) -> float:
    """Determine the turn direction going from a to b to c.

    Going from a->b->c, is the turn clockwise, counterclockwise, or straight.
       positive => CCW
       negative => CW
       zero => colinear

    See: https://algs4.cs.princeton.edu/91primitives/

    >>> a = Point(1,1)
    >>> b = Point(2,2)
    >>> turn(a, b, Point(3,2))
    -1
    >>> turn(a, b, Point(2,3))
    1
    >>> turn(a, b, Point(3,3))
    0
    >>> turn(a, b, Point(1.5, 1.5)) == 0
    True
    >>> turn(a, b, Point(1.5,1.7)) > 0
    True

    :param Point a:
    :param Point b:
    :param Point c:
    :rtype: float
    """
    return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)


if __name__ == "__main__":
    import doctest


    doctest.testmod()

# End of rocch.py
