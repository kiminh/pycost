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

Point: namedtuple = namedtuple( "Point", ("x", "y", "clfname"), defaults=(None,) )


class ROCCH( object ):
    """ROC Convex Hull.

    Some other stuff.
    """

    def __init__(self, keep_intermediate=False):
        """Initialize the object."""
        self.keep_intermediate = keep_intermediate
        self.classifiers: Dict[str, List[Tuple]] = { }
        self._hull = [Point( 0, 0, "AllNeg" ), Point( 1, 1, "AllPos" )]

    def fit(self, clfname: str, points):
        """Fit (add) a classifier's ROC points to the ROCCH.

        :param clfname: A classifier identifier.  This is only used to record the identity of the 
        classifier producing the points. It can be anything, such as a (classifier, threshold) 
        pair. 
        :param points: A sequence of ROC points, contained in a list or array.  Each point should
        be an (FP, TP) pair.  TODO: Make this more general.
        :return: None
        """
        points_instances = [Point( x, y, clfname ) for (x, y) in points]
        points_instances.extend( self._hull )
        points_instances.sort( key=lambda pt: pt.x )
        hull = []

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
                else:  # CW turn, this is convex
                    test_top = False
        self._hull = hull

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
        Returns a list consisting of (prob_min, prob_max, clf).

        :return: 
        :rtype: 
        """
        pass


def check_hull(hull):
    """Check a list of hull points for convexity.

    :param hull: A list of Point instances describing an ROC convex hull.
    :return: None 
    """
    while len( hull ) >= 3:
        assert turn( *hull[0:3] ) < 0, f"non-convex segment: {hull[0:3]}"
        hull.pop( 0 )


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
