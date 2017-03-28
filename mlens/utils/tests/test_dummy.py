"""ML-ENSEMBLE

"""

import numpy as np


from mlens.utils.dummy import OLS, LogisticRegression, Scale

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


def test_ols_estimators():
    """[Utils] OLS: check estimators."""
    if check_estimator is not None:
        check_estimator(OLS)


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


def test_logisticregression_estimators():
    """[Utils] LogisticRegression: check estimators"""
    if check_estimator is not None:
        check_estimator(LogisticRegression)


def test_logisticregression_weights():
    """[Utils] LogisticRegression: check weights."""
    w = np.array([[-0.81643357,  0.76923077],
                  [-0.52156177,  0.51282051],
                  [ 0.33799534, -0.28205128]])

    lr = LogisticRegression()
    lr.fit(X, z)

    np.testing.assert_array_almost_equal(lr.coef_, w)


def test_logisticregression_preds_labels():
    """[Utils] LogisticRegression: check label predictions."""
    g = np.array([1., 1., 1., 1., 2., 2., 2., 3., 3., 3., 3., 3.])

    p = LogisticRegression().fit(X, z).predict(X)

    np.testing.assert_array_equal(p, g)


def test_logisticregression_preds_labels():
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
