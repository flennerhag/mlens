"""ML-ENSEMBLE

author: Sebastian Flennerhag
"""

from __future__ import division, print_function

import numpy as np

from mlens.preprocessing.feature_engineering import PredictionFeature
from mlens.metrics.metrics import rmse_scoring
from mlens.utils.dummy import AverageRegressor


from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import check_estimator

import warnings

import os
try:
    from contextlib import redirect_stderr
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stderr


SEED = 100
np.random.seed(SEED)

# training data
X = np.random.random((100, 10))

# noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

# Change scales
X[:, 0] *= 10
X[:, 1] += 10
X[:, 2] *= 5
X[:, 3] *= 3
X[:, 4] /= 10

pf = PredictionFeature([Lasso(alpha=0.01),
                        RandomForestRegressor(random_state=SEED)],
                       scorer=rmse_scoring, verbose=1, random_state=SEED)


class Temp(PredictionFeature):
    """Temporary InitMixing for PreprocessingFeature.

    Until PreprocessingFeature is ported to new layer API, This Mixin is
    used to instantiate an estimator.
    """
    def __init__(self):
        super(Temp, self).__init__(estimators=[AverageRegressor()])


def test_estimator_behavior():
    """[PredictionFeature] Test Scikit-learn compatibility."""
    with warnings.catch_warnings(record=True) as w:
        # Assert the SuperLearner passes the Scikit-learn estimator test.
        check_estimator(Temp)

    # Check that all warnings were mlens warnings
    if not isinstance(w, list):
        w = [w]

    for m in w:
        filename = m.filename.lower()

        assert 'mlens' in filename
        assert 'sklearn' not in filename
        assert 'scikit-learn' not in filename


def test_prediction_feature_no_concat():
    """[PredictionFeature] Test run with concat = False."""
    pf.set_params(**{'concat': False})

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        pf.fit(X[:50], y[:50])
        Z = pf.transform(X[:50])

    assert Z.shape[1] == 2


def test_prediction_feature_concat():
    """[PredictionFeature] Test run with concat = True."""
    pf.set_params(**{'concat': True,
                     'verbose': 0})

    pf.fit(X[:50], y[:50])
    Z = pf.transform(X[:50])

    assert Z.shape[1] == 12


def test_prediction_feature_folds():
    """[PredictionFeature] Test use of folded predictions for training set."""
    pf.set_params(**{'concat': False,
                     'verbose': 0})

    # Check that fit and fit_transform give the same
    Z = pf.fit_transform(X, y)
    Z2 = pf.transform(X)

    # Check that separating train and test set works
    pf.fit(X[:50], y[:50])
    H = pf.transform(X[50:])

    H1 = pf.fit_transform(X[50:], y[50:])

    assert np.array_equal(Z, Z2)
    assert not np.array_equal(H, H1)


def test_prediction_feature_pipelining():
    """[PredictionFeature] Test in Scikit-learn pipeline object."""
    pf.set_params(**{'concat': True,
                     'folds': KFold(5),
                     'verbose': 0})

    pipe = make_pipeline(StandardScaler(), pf, Lasso(alpha=0.01))
    pipe.fit(X[:50], y[:50])
    p = pipe.predict(X[50:])

    assert str(rmse_scoring(y[50:], p)) == '0.160497386705'
