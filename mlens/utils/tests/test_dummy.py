"""ML-ENSEMBLE

"""

import numpy as np
import os
try:
    from contextlib import redirect_stdout
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stdout

from mlens.base import FoldIndex, SubsetIndex, BlendIndex
from mlens.ensemble.base import Layer, LayerContainer
from mlens.utils.dummy import OLS, LogisticRegression, Scale, InitMixin
from mlens.utils.dummy import ESTIMATORS, PREPROCESSING, ESTIMATORS_PROBA, \
    ECM, ECM_PROBA
from mlens.utils.dummy import get_layers, get_path, destroy_temp_dir, data, \
    ground_truth, _layer_est

from mlens.utils import assert_correct_layer_format
from mlens.utils.formatting import _assert_format

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
    assert_correct_layer_format(ESTIMATORS_PROBA, PREPROCESSING)
    assert_correct_layer_format(ESTIMATORS, PREPROCESSING)
    assert_correct_layer_format(ECM, None)
    assert_correct_layer_format(ECM_PROBA, None)

    assert _assert_format(ESTIMATORS)
    assert _assert_format(ESTIMATORS_PROBA)
    assert _assert_format(ECM)
    assert _assert_format(ECM_PROBA)
    assert _assert_format(PREPROCESSING)


def test_get_layers():
    """[Utils] testing: test dummy estimator and preprocessing formatting."""
    for p in [False, True]:
        for cls in ['stack', 'blend', 'subset']:
            layer, lc, lcm = get_layers(cls, p)

            assert isinstance(layer, Layer)
            assert isinstance(lc, LayerContainer)
            assert isinstance(lcm, LayerContainer)


def test_tmp_dir():
    """[Utils] testing: test tmp dir open and close."""
    path = get_path()
    assert os.path.exists(path)

    destroy_temp_dir(path)
    assert not os.path.exists(path)


def test_ground_truth():
    """[Utils] testing: test ground truth for stacking."""

    gf = np.array([[ 17.        ,  11.        , -42.        ],
                   [ 29.        ,  15.        , -30.        ],
                   [ 39.64705882,  17.64705882,  -6.35294118],
                   [ 52.35294118,  22.35294118,   6.35294118],
                   [ 63.        ,  25.        ,  30.        ],
                   [ 75.        ,  29.        ,  42.        ]])

    gwf = np.array([[ -5.        ,  11.        ],
                    [ -7.        ,   9.        ],
                    [ -1.52941176,   7.88235294],
                    [ -3.52941176,   5.88235294],
                    [ -3.        ,   9.        ],
                    [ -5.        ,   7.        ],
                    [  3.        ,   3.        ],
                    [  3.17647059,   3.17647059],
                    [  3.        ,   3.        ]])

    gp = np.array([[ 14.57142857,   8.57142857, -31.42857143],
                   [ 27.14285714,  13.14285714, -18.85714286],
                   [ 39.71428571,  17.71428571,  -6.28571429],
                   [ 52.28571429,  22.28571429,   6.28571429],
                   [ 64.85714286,  26.85714286,  18.85714286],
                   [ 77.42857143,  31.42857143,  31.42857143]])

    gwp = np.array([[-2.        ,  8.28571429],
                    [-4.        ,  6.28571429],
                    [ 3.14285714,  3.14285714]])


    t, z = data((6, 2), 2)

    indexer = FoldIndex(3, X=t)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        (F, wf), (P, wp) = ground_truth(t, z, indexer, 'predict', 1)

    np.testing.assert_array_almost_equal(F, gf)
    np.testing.assert_array_almost_equal(wf, gwf)
    np.testing.assert_array_almost_equal(P, gp)
    np.testing.assert_array_almost_equal(wp, gwp)


def test_layer_est():
    """[Utils] layer_estimation: testing layer estimation wrapper."""
    layer, _, _ = get_layers('stack', False)
    layer.indexer.fit(X)
    out = _layer_est(layer, 'fit', X, y, 1)

    assert isinstance(out, np.ndarray)
    assert out.shape[0] == X.shape[0]
    assert out.shape[1] == layer.n_pred
