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
import unittest

import numpy as np

# ML Ensemble
from mlens.ensemble import Ensemble
from mlens.model_selection import Evaluator
from mlens.metrics import rmse
from mlens.metrics.metrics import rmse_scoring
from mlens.ensemble._setup import name_estimators, name_base, _check_names
from mlens.ensemble._clone import _clone_base_estimators
from mlens.ensemble._clone import _clone_preprocess_cases
from mlens.parallel._preprocess_functions import _preprocess_pipe
from mlens.parallel._preprocess_functions import _preprocess_fold
from mlens.utils.utils import name_columns
from mlens.parallel import preprocess_folds, preprocess_pipes
# from mlens.metrics import score_matrix
# from mlens.parallel import fit_estimators, base_predict


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


def gen_base():
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
    return meta, base_pipelines


def gen_ensemble():

    meta, base_pipelines = gen_base()

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

    evals = Evaluator(X, y, rmse, preprocessing, cv=2, verbose=0,
                      shuffle=False, n_jobs_estimators=jobs,
                      n_jobs_preprocessing=jobs, random_state=100)

    return evals, estimators, parameters


# ===================== Test Class =====================
class TestClass(unittest.TestCase):

    def test_naming(self):
        np.random.seed(100)
        meta, base = gen_base()

        named_meta = name_estimators([meta], 'meta-')
        named_base = name_base(base)

        self.assertTrue(isinstance(named_meta, dict))
        self.assertTrue(isinstance(named_meta['meta-svr'], SVR))
        self.assertTrue(isinstance(named_base, dict))
        self.assertTrue(len(named_base) == 6)

    def test_check_names(self):
        np.random.seed(100)
        meta, base = gen_base()

        preprocess = [(case, _check_names(p[0])) for case, p in
                      base.items()]

        base_estimators = [(case, _check_names(p[1])) for case, p in
                           base.items()]

        self.assertTrue(isinstance(base_estimators, list))
        self.assertTrue(isinstance(preprocess, list))
        self.assertTrue(len(base_estimators) == 3)
        self.assertTrue(len(preprocess) == 3)
        self.assertTrue(isinstance(base_estimators[0], tuple))
        self.assertTrue(isinstance(preprocess[0], tuple))

    def test_clone(self):
        np.random.seed(100)
        meta, base = gen_base()

        preprocess = [(case, _check_names(p[0])) for case, p in
                      base.items()]
        base_estimators = [(case, _check_names(p[1])) for case, p in
                           base.items()]

        base_ = _clone_base_estimators(base_estimators)
        preprocess_ = _clone_preprocess_cases(preprocess)
        base_columns_ = name_columns(base_)

        self.assertTrue(isinstance(preprocess_, list))
        self.assertTrue(isinstance(preprocess_[0], tuple))
        self.assertTrue(isinstance(preprocess_[0][1], list))
        self.assertTrue(isinstance(base_, dict))
        self.assertTrue(isinstance(base_['mm'], list))
        self.assertTrue(isinstance(base_['mm'][0], tuple))
        self.assertTrue(isinstance(base_columns_, list))
        self.assertTrue(len(base_columns_) == 4)

    def test_preprocess_pipe_fun(self):
        np.random.seed(100)
        X, y = gen_data()

        out = _preprocess_pipe(X, y, X, [StandardScaler()], fit=True)
        self.assertTrue(out is not None)
        self.assertTrue(len(out) == 2)
        self.assertTrue(isinstance(out[0], np.ndarray))

        self.assertTrue(all([(_preprocess_pipe(X, y, X, [],
                                               fit=True)[i] == X).all() for
                             i in range(2)]))
        self.assertTrue((_preprocess_pipe(X, y, X, [],
                                          fit=True)[1] == X).all())

        out = _preprocess_pipe(X, y, None, [StandardScaler()], fit=True,
                               p_name='test')
        self.assertTrue(out[-1] == 'test')

    def test_preprocess_fold_fun(self):
        np.random.seed(100)
        X, y = gen_data()

        out = _preprocess_fold(X, y, (range(500), range(500, 1000)),
                               ('test', [StandardScaler()]), fit=True)
        self.assertTrue(out is not None)
        self.assertTrue(len(out) == 6)
        self.assertTrue(all([out[i].shape[1] == 10 for i in range(2)]))
        self.assertTrue(all([out[i].shape[0] == 500 for i in range(2, 4)]))
        self.assertTrue(out[5] == 'test')

    def test_preprocess_fold(self):
        np.random.seed(100)
        X, y = gen_data()
        meta, base = gen_base()

        preprocess = [(case, _check_names(p[0])) for case, p in
                      base.items()]

        data = preprocess_folds(_clone_preprocess_cases(preprocess),
                                X, y, folds=2, fit=True,
                                shuffle=False,
                                random_state=100,
                                n_jobs=-1, verbose=False)
        self.assertTrue(len(data) == 6)
        self.assertTrue(len(data[0]) == 6)
        self.assertTrue(all([isinstance(data[0][i], np.ndarray) for i in
                            range(4)]))
        self.assertTrue(all([data[0][i].shape == (500, 10) for i in
                            range(0, 2)]))
        self.assertTrue(all([data[0][i].shape == (500,) for i in range(2, 5)]))
        self.assertTrue(isinstance(data[0][-1], str))

        data = preprocess_folds([],
                                X, y, folds=2, fit=True,
                                shuffle=False,
                                random_state=100,
                                n_jobs=-1, verbose=False)
        self.assertTrue(data is not None)
        self.assertTrue(len(data) != 0)
        self.assertTrue((X[data[1][-1]] == data[0][0]).all())
        self.assertTrue((X[data[0][-1]] == data[1][0]).all())

    def test_preprocess_pipe(self):
        np.random.seed(100)
        X, y = gen_data()
        preprocess = [('test',
                       [StandardScaler(copy=True,
                                       with_mean=True, with_std=True)])]

        out = preprocess_pipes(preprocess, X, y, fit=True, dry_run=False,
                               return_estimators=True)

        pipes, Z, cases = zip(*out)
        preprocess_ = [(case, pipe) for case, pipe in
                       zip(cases, pipes)]

        self.assertTrue(isinstance(preprocess_, list))
        self.assertTrue(isinstance(preprocess_[0], tuple))
        self.assertTrue(preprocess_[0][0] == 'test')
        self.assertTrue(hasattr(preprocess_[0][1][0], 'mean_'))

    def test_grid_search(self):
        np.random.seed(100)
        X, y = gen_data()
        ensemble = gen_ensemble()
        ens_p = gen_params()

        grid = RandomizedSearchCV(ensemble, param_distributions=ens_p,
                                  n_iter=2, cv=2, scoring=rmse,
                                  n_jobs=-1, random_state=100)
        grid.fit(X, y)

        self.assertTrue(str(grid.best_score_)[:16] == '-0.0626352824626')

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

        score1 = rmse(ensemble, X[900:], y[900:])
        score2 = rmse_scoring(y[900:], ensemble.predict(X[900:]))

        self.assertTrue(score1 == -score2)
        self.assertTrue(ensemble.get_params()['n_jobs'] == -1)
        self.assertTrue(str(score1)[:16] == '-0.0522364178463')

    def test_evaluator(self):
        np.random.seed(100)
        X, y = gen_data()
        evals, ests, params = gen_eval(X, y, -1)

        evals.preprocess()

        evals.evaluate(ests, params, n_iter=2)

        self.assertTrue(str(evals.summary_.iloc[0, 0]) == '-0.357428210976')
