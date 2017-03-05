#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 15/01/2017
"""

from __future__ import division, print_function

import numpy as np

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from mlens.preprocessing.feature_engineering import PredictionFeature
from mlens.metrics.metrics import rmse_scoring


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

pf = PredictionFeature([Lasso(alpha=0.01), RandomForestRegressor(random_state=SEED)],
                       scorer=rmse_scoring, verbose=1, random_state=SEED)


def test_prediction_feature_no_concat():
    pf.set_params(**{'concat': False})

    pf.fit(X[:50], y[:50])
    Z = pf.transform(X[:50])

    assert Z.shape[1] == 2


def test_prediction_feature_concat():
    pf.set_params(**{'concat': True,
                     'verbose': 0})

    pf.fit(X[:50], y[:50])
    Z = pf.transform(X[:50])

    assert Z.shape[1] == 12


def test_prediction_feature_folds():
    pf.set_params(**{'concat': False,
                     'verbose': 0})

    pf.fit(X, y)
    Z = pf.transform(X)[50:]

    pf.fit(X, y)
    Z2 = pf.transform(X)[50:]

    pf.fit(X[:50], y[:50])
    H = pf.transform(X[50:])

    assert np.array_equal(Z, Z2)
    assert not np.array_equal(H, Z)


def test_prediction_feature_pipelining():
    pf.set_params(**{'concat': True,
                     'folds': KFold(5),
                     'verbose': 0})

    pipe = make_pipeline(StandardScaler(), pf, Lasso(alpha=0.01))
    pipe.fit(X[:50], y[:50])
    p = pipe.predict(X[50:])

    assert str(rmse_scoring(y[50:], p)) == '0.160497386705'
