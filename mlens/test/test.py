#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 10/01/2017
Simple test run of Ensemble
"""
import argparse
import numpy as np
from time import time

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

# ===================== Arguments =====================
msg = "A makesshift test to see that that it runs and gives good results"

parser = argparse.ArgumentParser(description=msg)

parser.add_argument('-g', help='bool : whether to test \
                    embedding an ensemble instance in an sklearn grid search',
                    required=False, type=bool, default=True)
parser.add_argument('-n', help='int : n_jobs. Set to -1, 1 or 2 (2 for both)',
                    required=False, type=int, default=-1)

args = parser.parse_args()


# ===================== Seed =====================
np.random.seed(100)  # doesn't seem to work for n_jobs=-1


# ===================== Setup functions =====================
def gen_data(size=1000):

    # training data
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


def gen_eval(X, y, scoring, jobs):

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

    evals = Evaluator(X, y, preprocessing, scoring, cv=2, verbose=0,
                      shuffle=False, n_jobs_estimators=jobs,
                      n_jobs_preprocessing=jobs, random_state=100)

    return evals, estimators, parameters


# ===================== Diagnostic functions =====================
def test_grid_search(ensemble, ens_p, X, y, jobs):

    grid = RandomizedSearchCV(ensemble, param_distributions=ens_p,
                              n_iter=2, cv=2, scoring=rmse, n_jobs=jobs,
                              random_state=100)
    grid.fit(X, y)

    assert str(grid.best_score_)[:16] == '-0.0626352824626'


def test_fit_predict(ensemble, X, y, jobs):

    # Test sklearn set params API
    ensemble.set_params(**{'np-rf__min_samples_leaf': 9,
                           'meta-svr__C': 16.626146983723014,
                           'sc-kn__n_neighbors': 9,
                           'np-rf__max_features': 4,
                           'mm-svr__C': 11.834807428701293,
                           'sc-ls__alpha': 0.0014284293508642438,
                           'np-rf__max_depth': 4,
                           'n_jobs': jobs,
                           'verbose': 0,
                           'shuffle': False,
                           'random_state': 100})

    ensemble.fit(X[:900], y[:900])

    score = rmse(ensemble, X[900:], y[900:])

    assert str(score)[:16] == '-0.0522364178463'


def test_evaluator(X, y, scoring, jobs):

    evals, ests, params = gen_eval(X, y, scoring, jobs)

    evals.preprocess()

    evals.evaluate(ests, params, n_iter=2)

    assert str(evals.summary_.iloc[0, 0]) == '-0.357428210976'


def main(args):
    print('-'*30, '\nBuild test of ML-Ensemble')
    print('Commencing test...')
    t0 = time()

    X, y = gen_data()
    ensemble = gen_ensemble()

    if args.n == 2:
        cases = [1, -1]
    else:
        cases = [args.n]

    if args.g:
        ens_p = gen_params()

        print('')
        for i, n in enumerate(cases):
            test_grid_search(ensemble, ens_p, X, y, n)
            print('Gridsearch test [%i/%i]: ok' % (i + 1, len(cases)))

    for i, n in enumerate(cases):
        test_fit_predict(ensemble, X, y, n)
        print('Ensemble test [%i/%i]: ok' % (i + 1, len(cases)))

    for i, n in enumerate(cases):
        test_evaluator(X, y, rmse, n)
        print('Evaluator test [%i/%i]: ok' % (i + 1, len(cases)))

    print('')
    print('Test completed in %.2f seconds' % (time() - t0))
    print('Build ok')

if __name__ == "__main__":
    main(args)
