#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sebastian Flennerhag
@date: 12/01/2017
"""

from mlens.model_selection import Evaluator
from mlens.metrics import rmse
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

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
                  shuffle=False, n_jobs_estimators=-1,
                  n_jobs_preprocessing=-1, random_state=100)


def test_evals():

        evals.preprocess()
        evals.evaluate(estimators, parameters, n_iter=2)

        assert str(evals.summary_.iloc[0, 0]) == '-0.357428210976'
