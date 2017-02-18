#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sebastian Flennerhag
@date: 12/01/2017
"""

from __future__ import division, print_function, with_statement

from mlens.ensemble import StackingEnsemble
from mlens.metrics import rmse
from mlens.metrics.metrics import rmse_scoring
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform, randint
import numpy as np
from pandas import DataFrame
import warnings

# training data
np.random.seed(100)
X = np.random.random((1000, 10))

# noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

# Change scales
X[:, 0] *= 10
X[:, 1] += 10
X[:, 2] *= 5
X[:, 3] *= 3
X[:, 4] /= 10

# meta estimator
meta = SVR()

# Create ensemble with preprocessing pipelines
base_pipelines = {'sc':
                  ([StandardScaler()],
                   [('ls', Lasso()), ('kn', KNeighborsRegressor())]),
                  'mm':
                  ([MinMaxScaler()], [SVR()]),
                  'np':
                  ([], [('rf', RandomForestRegressor(random_state=100))])}

ensemble = StackingEnsemble(folds=10, shuffle=False,
                            scorer=rmse._score_func, n_jobs=1,
                            random_state=100)
ensemble.add(base_pipelines)
ensemble.add_meta(meta)

params = {'layer-1-sc-ls__alpha': uniform(0.0005, 0.005),
          'layer-1-np-rf__max_depth': randint(2, 6),
          'layer-1-np-rf__max_features': randint(2, 5),
          'layer-1-np-rf__min_samples_leaf': randint(5, 12),
          'layer-1-sc-kn__n_neighbors': randint(6, 12),
          'layer-1-mm-svr__C': uniform(10, 20),
          'meta-svr__C': uniform(10, 20)}

# Ensembles without preprocessing
ens1 = StackingEnsemble(folds=KFold(2, random_state=100, shuffle=True),
                        random_state=100, n_jobs=-1, scorer=rmse_scoring)
ens1.add([('svr', SVR()), ('rf', RandomForestRegressor(random_state=100))],)
ens1.add_meta(Lasso(alpha=0.001, random_state=100))

ens2 = StackingEnsemble(random_state=100, n_jobs=-1)
ens2.add_meta(Lasso(alpha=0.001, random_state=100))
ens2.add([SVR(), RandomForestRegressor(random_state=100)])

ens3 = StackingEnsemble(random_state=100, n_jobs=-1)
ens3.add({'no_prep': ([], [SVR(), RandomForestRegressor(random_state=100)])})
ens3.add_meta(Lasso(alpha=0.001, random_state=100))


def test_no_preprocess_ensemble():
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

    assert str(ens1.scores_['layer-1-rf'])[:16] == '0.171221793126'
    assert str(ens1.scores_['layer-1-svr'])[:16] == '0.171500953229'


def test_preprocess_ensemble():

    ensemble.set_params(**{'layer-1-np-rf__min_samples_leaf': 9,
                           'meta-svr__C': 16.626146983723014,
                           'layer-1-sc-kn__n_neighbors': 9,
                           'layer-1-np-rf__max_features': 4,
                           'layer-1-mm-svr__C': 11.834807428701293,
                           'layer-1-sc-ls__alpha': 0.0014284293508642438,
                           'layer-1-np-rf__max_depth': 4,
                           'n_jobs': -1,
                           'verbose': 2,
                           'shuffle': False,
                           'random_state': 100})

    ensemble.fit(X[:900], y[:900])

    score1 = rmse(ensemble, X[900:], y[900:])
    score2 = rmse_scoring(y[900:], ensemble.predict(X[900:]))

    assert score1 == -score2
    assert ensemble.get_params()['n_jobs'] == -1
    assert str(score1)[:16] == '-0.0522364178463'


def test_grid_search():

    ensemble.set_params(**{'n_jobs': 1})

    grid = RandomizedSearchCV(ensemble, param_distributions=params,
                              n_iter=2, cv=2, scoring=rmse,
                              n_jobs=-1, random_state=100)
    grid.fit(X, y)
    print(grid.best_score_)
    assert str(grid.best_score_)[:16] == '-0.0626352824626'


def test_ensemble_exception_handling():
    # Currently this test just ensures the ensemble runs through
    ensemble.set_params(**{  # will cause error
                           'layer-1-np-rf__min_samples_leaf': 1.01,
                           'meta-svr__C': 1,
                           'layer-1-sc-kn__n_neighbors': 2,
                             # will cause error
                           'layer-1-np-rf__max_features': 1.01,
                           'layer-1-mm-svr__C': 11.834807428701293,
                           'layer-1-sc-ls__alpha': 0.0014284293508642438,
                           'layer-1-np-rf__max_depth': 4,
                           'n_jobs': -1,
                           'verbose': 0,
                           'shuffle': False,
                           'random_state': 100})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble.fit(X[:100], y[:100])
