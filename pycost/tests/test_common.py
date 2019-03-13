import pytest

from sklearn.utils.estimator_checks import check_estimator

from pycost import TemplateEstimator
from pycost import TemplateClassifier
from pycost import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
