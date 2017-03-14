"""ML-ENSEMBLE

author: Sebastian Flennerhag
"""

from __future__ import division, print_function, with_statement

import numpy as np
from pandas import DataFrame

from mlens.ensemble import StackingEnsemble
from mlens.metrics import rmse
from mlens.metrics.metrics import rmse_scoring
from mlens.utils.estimator_checks import InitMixin

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

import warnings

import os
try:
    from contextlib import redirect_stderr
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stderr


# Training data
np.random.seed(100)
X = np.random.random((1000, 10))

# Noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

# Change scales
X[:, 0] *= 10
X[:, 1] += 10
X[:, 2] *= 5
X[:, 3] *= 3
X[:, 4] /= 10

fit_params = {'layers__layer-1__np-rf__min_samples_leaf': 9,
              'meta_estimator__C': 16.626146983723014,
              'layers__layer-1__sc-kn__n_neighbors': 9,
              'layers__layer-1__np-rf__max_features': 4,
              'layers__layer-1__mm-svr__C': 11.834807428701293,
              'layers__layer-1__sc-ls__alpha': 0.0014284293508642438,
              'layers__layer-1__np-rf__max_depth': 4,
              'n_jobs': -1,
              'verbose': 2,
              'shuffle': False,
              'random_state': 100}

grid_params = {'layers__layer-1__mm-svr__C': [3, 5],
               'meta_estimator__C': [1, 2]}

# Complex ensemble
meta = SVR()
preprocessing = {'sc': [StandardScaler()],
                 'mm': [MinMaxScaler()],
                 'np': []}
estimators = {'sc': [('ls', Lasso()), ('kn', KNeighborsRegressor())],
              'mm': [SVR()],
              'np': [('rf', RandomForestRegressor(random_state=100))]}

ensemble = StackingEnsemble(folds=10, shuffle=False,
                            scorer=rmse._score_func, n_jobs=1,
                            random_state=100)
ensemble.add(preprocessing=preprocessing, estimators=estimators)
ensemble.add_meta(meta)


# Ensembles without preprocessing to check case handling and estimator perf
# Named estimator tuples, implicit lack of preprocessing
ens1 = StackingEnsemble(folds=KFold(2, random_state=100, shuffle=True),
                        random_state=100, n_jobs=-1, scorer=rmse_scoring)
ens1.add([('svr', SVR()), ('rf', RandomForestRegressor(random_state=100))])
ens1.add_meta(Lasso(alpha=0.001, random_state=100))

# Ensemble without named tuples, implicit lack of preprocessing
ens2 = StackingEnsemble(random_state=100, n_jobs=-1)
ens2.add_meta(Lasso(alpha=0.001, random_state=100))
ens2.add([SVR(), RandomForestRegressor(random_state=100)])

# Ensemble with explicit no prep pipelines
ens3 = StackingEnsemble(random_state=100, n_jobs=-1)
ens3.add({'no_prep': [SVR(), RandomForestRegressor(random_state=100)]},
         {'no_prep': []})

ens3.add_meta(Lasso(alpha=0.001, random_state=100))

# Simple ensemble for grid search

grid_ens = StackingEnsemble(folds=5, shuffle=False,
                            scorer=rmse_scoring, n_jobs=1,
                            random_state=100)
grid_ens.add({'mm': [SVR()]}, {'mm': [MinMaxScaler()]})
grid_ens.add_meta(SVR())


class TestStackingEnsemble(InitMixin, StackingEnsemble):
    """Wrapper around StackingEnsemble to check estimator behavior."""

    def __init__(self):
        super(TestStackingEnsemble, self).__init__()


def test_estimator_behavior():
    """[StackingEnsemble] Test Scikit-learn compatibility."""
    with warnings.catch_warnings(record=True) as w:
        # Assert the StackingEnsemble passes the Scikit-learn estimator test.
        check_estimator(TestStackingEnsemble)

    # Check that all warnings were mlens warnings
    if not isinstance(w, list):
        w = [w]

    for m in w:
        filename = m.filename.lower()

        assert 'mlens' in filename
        assert 'sklearn' not in filename
        assert 'scikit-learn' not in filename


def test_no_preprocess_ensemble():
    """[StackingEnsemble] Test without any preprocessing."""
    ens1.fit(X, y)
    ens2.fit(X, y)
    ens3.fit(X, y)

    p1 = ens1.predict(X)
    p2 = ens2.predict(X)
    p3 = ens3.predict(X)

    assert (p1 == p2).all()
    assert (p1 == p3).all()

    # Test that DataFrames runs through
    ens1.set_params(**{'as_df': True, 'scorer': rmse._score_func})
    ens1.fit(DataFrame(X), y)

    assert str(ens1.scores_['layer-1--rf'])[:16] == '0.171221793126'
    assert str(ens1.scores_['layer-1--svr'])[:16] == '0.171500953229'


def test_preprocess_ensemble():
    """[StackingEnsemble] Test with preprocessing."""
    # Check that cloning and set params work
    ens = clone(ensemble).set_params(**fit_params)

    ensemble.set_params(**fit_params)

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        # Silences the training output, but lets warnings and errors through.
        ens.fit(X[:900], y[:900])
        ensemble.fit(X[:900], y[:900])

        score1 = rmse(ens, X[900:], y[900:])
        score2 = rmse_scoring(y[900:], ens.predict(X[900:]))

    assert score1 == -score2
    assert ens.get_params()['n_jobs'] == -1
    assert str(score1)[:16] == '-0.0522364178463'


def test_grid_search():
    """[StackingEnsemble] Test in GridSearch."""
    grid_ens.set_params(**{'n_jobs': 1})

    grid = GridSearchCV(grid_ens, param_grid=grid_params,
                        cv=2, scoring=rmse, n_jobs=1)

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        # Silences the training output, but lets warnings and errors through.
        grid.fit(X, y)
    assert str(grid.best_score_)[:16] == '-0.0739108130567'


def test_ensemble_exception_handling():
    """[StackingEnsemble] Test exception handling."""
    # Currently this test just ensures the ensemble runs through
    ensemble.set_params(**{  # will cause error
                           'layers__layer-1__np-rf__min_samples_leaf': 1.01,
                           'meta_estimator__C': 1,
                           'layers__layer-1__sc-kn__n_neighbors': 2,
                             # will cause error
                           'layers__layer-1__np-rf__max_features': 1.01,
                           'layers__layer-1__mm-svr__C': 11.834807428701293,
                           'layers__layer-1__sc-ls__alpha':
                           0.0014284293508642438,
                           'layers__layer-1__np-rf__max_depth': 4,
                           'n_jobs': -1,
                           'verbose': 0,
                           'shuffle': False,
                           'random_state': 100})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble.fit(X[:100], y[:100])
