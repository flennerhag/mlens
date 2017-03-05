"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 12/01/2017
"""

from __future__ import division, print_function

from mlens.model_selection import Evaluator, EnsembleLayers
from mlens.metrics import rmse
from mlens.utils import pickle_save, pickle_load
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import uniform, randint
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

# A set of estimators to evaluate
ls = Lasso(random_state=100)
rf = RandomForestRegressor(random_state=100)

# Some parameter distributions that might work well
ls_p = {'alpha': uniform(0.00001, 0.0005)}
rf_p = {'max_depth': randint(2, 7), 'max_features': randint(3, 10),
        'min_samples_leaf': randint(2, 10)}
rf_p_e = {'min_samples_leaf': uniform(1.01, 1.05)}

# Put it all in neat dictionaries. Note that the keys must match!
estimators = {'ls': ls, 'rf': rf}
parameters = {'ls': ls_p, 'rf': rf_p}
parameters_exception = {'ls': ls_p, 'rf': rf_p_e}

# A set of different preprocessing cases we want to try for each model
preprocessing = {'a': [StandardScaler()],
                 'b': []}

evals1 = Evaluator(rmse, preprocessing, cv=KFold(2, random_state=100),
                   verbose=1, shuffle=False, n_jobs_estimators=-1,
                   n_jobs_preprocessing=-1, random_state=100)

evals2 = Evaluator(rmse, preprocessing, cv=2,
                   verbose=1, shuffle=False, n_jobs_estimators=-1,
                   n_jobs_preprocessing=-1, random_state=100)

ens_base = EnsembleLayers()
ens_base.add([(key, val) for key, val in estimators.items()])

def check_scores(evals):
    test_draws = []
    for params in evals.cv_results_.loc[[('rf-a', 1),
                                         ('rf-a', 2),
                                         ('rf-a', 3)], 'params'].values:
        test_draws.append(params['min_samples_leaf'])

    assert all([test_val == comp_val for test_val, comp_val in
                zip(test_draws, [2, 2, 5])])

    assert str(evals.summary_.iloc[0][0])[:16] == '-0.127723982162'


def test_evals():
        evals1.preprocess(X, y)
        evals1.evaluate(X, y, estimators, parameters, 3, flush_preprocess=True)
        evals1.evaluate(X, y, estimators, parameters, 3)
        check_scores(evals1)


def test_pickling_evals():
        evals2.evaluate(X, y, estimators, parameters, 3)
        pickle_save(evals2, 'test')
        pickled_eval = pickle_load('test')
        check_scores(pickled_eval)


def test_exception_handling_evals():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evals1.evaluate(X, y, estimators, parameters_exception, 3)
        assert str(evals1.summary_.iloc[-1][0])[:3] == '-99'


def test_ensemble_layers():
    ens_base.fit(X, y)
    out = ens_base.transform(X)

    assert out.shape[1] == 2
