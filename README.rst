.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/pycost.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/pycost

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/pycost

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/pycost/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/pycost

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/pycost.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/pycost/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/pycost/badge/?version=latest
.. _ReadTheDocs: https://pycost.readthedocs.io/en/latest/?badge=latest

pycost - Code for classifier evaluation
============================================================

.. _scikit-learn: https://scikit-learn.org
.. _sklearn-metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
.. _imbalanced-learn: https://github.com/scikit-learn-contrib/imbalanced-learn

**pycost** is a scikit-learn_ compatible extension providing various evaluation-related
classes and functions commonly used in cost-sensitive and imbalanced-class classification.
Cost-sensitive learning generally refers to situations in which certain kinds of errors (e.g. false positives or false negatives) are more expensive than others.  In such cases many common metrics such as accuracy and the F1 metric may be misleading.  Imbalanced (or skewed) class learning refers to situations in which classes are not equally represented.  Cost-sensitive and imbalanced learning are closely related.

Pycost is a contributed package and not a part of scikit-learn_.  It may be seen as an adjunct to sklearn-metrics_.  Specifically, it includes code for:

- Computing and describing the ROC Convex Hull
- Computing and describing cost curves
- Computing the Area Under the ROC Curve (AUC) for multi-class (>2 classes) problems.
- Averaging ROC curves.
- TODO: Graphing the confidence intervals of ROC curves.

.. _documentation: https://pycost.readthedocs.io/en/latest/quick_start.html

Refer to the documentation_ for details.

This package is primarily for *evaluation* of classifiers.  If you're searching for techniques for learning with such datasets, try the fine imbalanced-learn_ package.
