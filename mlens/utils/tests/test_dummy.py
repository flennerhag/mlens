"""ML-ENSEMBLE

"""


import numpy as np
import os
try:
    from contextlib import redirect_stdout
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stdout

from mlens.ensemble.base import Layer, Sequential
from mlens.utils.dummy import OLS, LogisticRegression, Scale
from mlens.testing.dummy import ESTIMATORS, PREPROCESSING, ESTIMATORS_PROBA, \
    ECM, ECM_PROBA, InitMixin
from mlens.testing.dummy import Data, EstimatorContainer

from mlens.utils import assert_correct_format
from mlens.utils.formatting import _assert_format, _format_instances

from mlens.utils.exceptions import NotFittedError
from mlens.externals.sklearn.base import BaseEstimator

try:
    from sklearn.utils.estimator_checks import check_estimator
except:
    check_estimator = None


X = np.arange(24).reshape(12, 2)

# Regression
y = X[:, 0] * 3 - X[:, 1] * 0.5

# Classification
z = np.ones(12)
z[:3] = 1
z[3:8] = 2
z[8:12] = 3


if check_estimator is not None:

    class Ens(BaseEstimator):

        """Dummy ensemble class for testing InitMixin."""

        def __init__(self, layers=None):
            self.layers = layers

        def add(self, e):
            """Pass through."""
            return self

        def add_meta(self, e):
            """Add estimator."""
            self.layers = e
            return self

        def fit(self, X, y):
            """Call fit on estimator."""
            self.layers.fit(X, y)
            return self

        def predict(self, X):
            """Call predict on estimator."""
            return self.layers.predict(X)


    class TestMixin(InitMixin, Ens):

        """Simple test class of dummy ensemble."""

        def __init__(self):
            super(TestMixin, self).__init__()


def test_ols_estimators():
    """[Utils] OLS: check estimators."""
    if check_estimator is not None:
        check_estimator(OLS)


def test_ols_not_fitted():
    """[Utils] LogisticRegression: check raises if not fitted."""
    np.testing.assert_raises(NotFittedError, OLS().predict, X)


def test_ols_weights():
    """[Utils] OLS: check weights."""
    ols = OLS()
    ols.fit(X, y)

    np.testing.assert_array_almost_equal(ols.coef_, np.array([3., -0.5]))


def test_ols_preds():
    """[Utils] OLS: check predictions."""
    g = np.array([ 29.5,  34.5,  39.5,  44.5,  49.5,  54.5])

    p = OLS().fit(X[:6], y[:6]).predict(X[6:])

    np.testing.assert_array_almost_equal(p, g)


def test_logistic_regression_estimators():
    """[Utils] LogisticRegression: check estimators."""
    if check_estimator is not None:
        check_estimator(LogisticRegression)


def test_logistic_regression_not_fitted():
    """[Utils] LogisticRegression: check raises if not fitted."""
    np.testing.assert_raises(NotFittedError, LogisticRegression().predict, X)


def test_logistic_regression_weights():
    """[Utils] LogisticRegression: check weights."""
    w = np.array([[-0.81643357,  0.76923077],
                  [-0.52156177,  0.51282051],
                  [ 0.33799534, -0.28205128]])

    lr = LogisticRegression()
    lr.fit(X, z)

    np.testing.assert_array_almost_equal(lr.coef_, w)


def test_logistic_regression_preds_labels():
    """[Utils] LogisticRegression: check label predictions."""
    g = np.array([1., 1., 1., 1., 2., 2., 2., 3., 3., 3., 3., 3.])

    p = LogisticRegression().fit(X, z).predict(X)

    np.testing.assert_array_equal(p, g)


def test_logistic_regression_preds_proba():
    """[Utils] LogisticRegression: check label predictions."""

    g = np.array([[0.68335447, 0.62546744, 0.42995095],
                  [0.66258275, 0.62136312, 0.45756156],
                  [0.64116395, 0.61724135, 0.48543536],
                  [0.61916698, 0.61310265, 0.51340005],
                  [0.59666983, 0.60894755, 0.54128111],
                  [0.57375858, 0.60477659, 0.56890605],
                  [0.55052625, 0.60059033, 0.59610873],
                  [0.5270714, 0.59638931, 0.62273319],
                  [0.50349645, 0.59217411, 0.64863706],
                  [0.47990593, 0.58794531, 0.67369429],
                  [0.45640469, 0.58370348, 0.69779712],
                  [0.43309595, 0.57944923, 0.72085727]])

    p = LogisticRegression().fit(X, z).predict_proba(X)

    np.testing.assert_array_almost_equal(p, g)


def test_scale_estimators():
    """[Utils] Scale: check estimators"""
    if check_estimator is not None:
        check_estimator(Scale)


def test_scale_transformation():
    """[Utils] Scale: check transformation."""
    g = np.array([[-2., -4.],
                  [0., 0.],
                  [2., 4.]])

    x = np.arange(6).reshape(3, 2)
    x[:, 1] *= 2
    s = Scale().fit_transform(x)

    np.testing.assert_array_equal(s, g)


def test_scale_not_fitted():
    """[Utils] Scale: check not fitted."""

    np.testing.assert_raises(NotFittedError, Scale().transform, X)


def test_init_mixin():
    """[Utils] InitMixin: test mixin."""
    if check_estimator is not None:
        check_estimator(TestMixin)


def test_estimator_lists():
    """[Utils] testing: test dummy estimator and preprocessing formatting."""
    assert_correct_format(ESTIMATORS_PROBA, PREPROCESSING)
    assert_correct_format(ESTIMATORS, PREPROCESSING)
    assert_correct_format(ECM, None)
    assert_correct_format(ECM_PROBA, None)

    assert _assert_format(_format_instances(ESTIMATORS))
    assert _assert_format(_format_instances(ESTIMATORS_PROBA))
    assert _assert_format(ECM)
    assert _assert_format(ECM_PROBA)
    assert _assert_format(PREPROCESSING)


def test_get_layers():
    """[Utils] testing: test dummy estimator and preprocessing formatting."""
    for p in [False, True]:
        for cls in ['stack', 'blend', 'subsemble']:
            layer = EstimatorContainer().get_layer(cls, p, True)
            lc = EstimatorContainer().get_sequential(cls, p, True)

            assert isinstance(layer, Layer)
            assert isinstance(lc, Sequential)


def test_ground_truth():
    """[Utils] testing: test ground truth for stacking."""

    gf = np.array([[ 11.        ,  17.        , -14.        , -42.        ],
                   [ 15.        ,  29.        , -10.        , -30.        ],
                   [ 17.64705882,  39.64705882,  -2.35294118,  -6.35294118],
                   [ 22.35294118,  52.35294118,   2.35294118,   6.35294118],
                   [ 25.        ,  63.        ,  10.        ,  30.        ],
                   [ 29.        ,  75.        ,  14.        ,  42.        ]])

    gwf = np.array([[ -7.        ,   9.        ],
                   [ -3.52941176,   5.88235294],
                   [ -5.        ,   7.        ],
                   [ -5.        ,  11.        ],
                   [ -1.52941176,   7.88235294],
                   [ -3.        ,   9.        ],
                   [  1.        ,   1.        ],
                   [  1.17647059,   1.17647059],
                   [  1.        ,   1.        ],
                   [  3.        ,   3.        ],
                   [  3.17647059,   3.17647059],
                   [  3.        ,   3.        ]])


    gp = np.array([[  8.57142857,  14.57142857, -11.42857143, -31.42857143],
                   [ 13.14285714,  27.14285714,  -6.85714286, -18.85714286],
                   [ 17.71428571,  39.71428571,  -2.28571429,  -6.28571429],
                   [ 22.28571429,  52.28571429,   2.28571429,   6.28571429],
                   [ 26.85714286,  64.85714286,   6.85714286,  18.85714286],
                   [ 31.42857143,  77.42857143,  11.42857143,  31.42857143]])

    gwp = np.array([[-4.        ,  6.28571429],
                    [-2.        ,  8.28571429],
                    [1.14285714 ,  1.14285714],
                    [3.14285714 ,  3.14285714]])

    data = Data('stack', False, True, folds=3)
    t, z = data.get_data((6, 2), 2)
    (F, wf), (P, wp) = data.ground_truth(t, z)

    np.testing.assert_array_almost_equal(F, gf)
    np.testing.assert_array_almost_equal(wf, gwf)
    np.testing.assert_array_almost_equal(P, gp)
    np.testing.assert_array_almost_equal(wp, gwp)
