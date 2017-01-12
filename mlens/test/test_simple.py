#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 10/01/2017
An initial test suite for ML-Ensemble. The test covers the Ensemble class,
both as integrated with grid search and stand-alone, and the Evaluator class.
The test relies on fitting a dummy problem using a fixed seed and ensuring
that the optimized estimator / ensemble gives the right score. Since the
problem is deterministic, if the ensemble finds another score, the
learning algorithm has changed.
"""
import numpy as np

# ML Ensemble
from mlens.ensemble import Ensemble
from mlens.model_selection import Evaluator
from mlens.metrics import rmse

# Base Models
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# CV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


# ===================== Setup functions =====================
def gen_data(size=1000):

    # training data
    np.random.seed(100)
    X = np.random.random((size, 10))

    # noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
    y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

    # Change scales
    X[:, 0] *= 10
    X[:, 1] += 10
    X[:, 2] *= 5
    X[:, 3] *= 3
    X[:, 4] /= 10

    return X, y


def gen_ensemble():

    # meta estimator
    meta = SVR()

    # Create base estimators, along with associated preprocessing pipelines
    base_pipelines = {'sc':
                      ([StandardScaler()],
                       [('ls', Lasso()), ('kn', KNeighborsRegressor())]),
                      'mm':
                      ([MinMaxScaler()], [SVR()]),
                      'np':
                      ([], [('rf', RandomForestRegressor(random_state=100))])}

    ensemble = Ensemble(meta, base_pipelines, folds=10, shuffle=False,
                        scorer=rmse._score_func, n_jobs=1, random_state=100)
    return ensemble


def gen_params():
    return {'sc-ls__alpha': uniform(0.0005, 0.005),
            'np-rf__max_depth': randint(2, 6),
            'np-rf__max_features': randint(2, 5),
            'np-rf__min_samples_leaf': randint(5, 12),
            'sc-kn__n_neighbors': randint(6, 12),
            'mm-svr__C': uniform(10, 20),
            'meta-svr__C': uniform(10, 20)}


def gen_eval(X, y, jobs):
    np.random.seed(100)
    # A set of estimators to evaluate
    ls = Lasso(random_state=100)
    rf = RandomForestRegressor(random_state=100)

    # Some parameter distributions that might work well
    ls_p = {'alpha': uniform(0.0005, 10)}
    rf_p = {'max_depth': randint(2, 7), 'max_features': randint(3, 10),
            'min_samples_leaf': randint(2, 10)}

    # Put it all in neat dictionaries. Note that the keys must match!
    estimators = {'ls': ls, 'rf': rf}
    parameters = {'ls': ls_p, 'rf': rf_p}

    # A set of different preprocessing cases we want to try for each model
    preprocessing = {'a': [StandardScaler()],
                     'b': []}

    evals = Evaluator(X, y, preprocessing, rmse, cv=2, verbose=0,
                      shuffle=False, n_jobs_estimators=jobs,
                      n_jobs_preprocessing=jobs, random_state=100)

    return evals, estimators, parameters


# ===================== Test Class =====================
class TestClass(object):

    def test_grid_search(self):
        np.random.seed(100)
        X, y = gen_data()
        ensemble = gen_ensemble()
        ens_p = gen_params()

        grid = RandomizedSearchCV(ensemble, param_distributions=ens_p,
                                  n_iter=2, cv=2, scoring=rmse,
                                  n_jobs=-1, random_state=100)
        grid.fit(X, y)

        assert str(grid.best_score_)[:16] == '-0.0626352824626'

    def test_fit_predict(self):
        np.random.seed(100)
        X, y = gen_data()
        ensemble = gen_ensemble()

        # Test sklearn set params API
        ensemble.set_params(**{'np-rf__min_samples_leaf': 9,
                               'meta-svr__C': 16.626146983723014,
                               'sc-kn__n_neighbors': 9,
                               'np-rf__max_features': 4,
                               'mm-svr__C': 11.834807428701293,
                               'sc-ls__alpha': 0.0014284293508642438,
                               'np-rf__max_depth': 4,
                               'n_jobs': -1,
                               'verbose': 0,
                               'shuffle': False,
                               'random_state': 100})

        ensemble.fit(X[:900], y[:900])

        score = rmse(ensemble, X[900:], y[900:])

        assert ensemble.n_jobs == -1
        assert str(score)[:16] == '-0.0522364178463'

    def test_evaluator(self):
        np.random.seed(100)
        X, y = gen_data()
        evals, ests, params = gen_eval(X, y, -1)

        evals.preprocess()

        evals.evaluate(ests, params, n_iter=2)

        assert str(evals.summary_.iloc[0, 0]) == '-0.357428210976'
