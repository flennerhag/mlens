"""ML-ENSEMBLE

:author: Sebastian Flennerhag
"""

from __future__ import division, print_function

import numpy as np

from mlens.model_selection import Evaluator
from mlens.metrics import rmse
from mlens.utils import pickle_save, pickle_load
from mlens.utils.exceptions import FitFailedError, FitFailedWarning
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import uniform, randint

import warnings
import subprocess

import os
try:
    from contextlib import redirect_stderr
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stderr


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
estimators = [('ls', ls), ('rf', rf)]
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


def check_scores(evals):
    """Check that a fitted Evaluator has found the right scores."""
    test_draws = []
    for params in evals.cv_results_.loc[[('rf-a', 1),
                                         ('rf-a', 2),
                                         ('rf-a', 3)], 'params'].values:
        test_draws.append(params['min_samples_leaf'])

    assert all([test_val == comp_val for test_val, comp_val in
                zip(test_draws, [2, 2, 5])])

    assert str(evals.summary_.iloc[0][0])[:16] == '-0.127723982162'


def test_evals_preprocessing():
    """[Evaluator] Check preprocessing."""
    with open(os.devnull, 'w') as f, redirect_stderr(f):
        # Silences the training output, but lets warnings and errors through.
        evals1.preprocess(X, y)


def test_evals_evaluate_preprocessed():
    """[Evaluator] Check evaluate on preprocessed folds."""
    with open(os.devnull, 'w') as f, redirect_stderr(f):
        # Silences the training output, but lets warnings and errors through.
        evals1.evaluate(estimators, parameters, X, y, 3)

    check_scores(evals1)


def test_evals_evaluate():
    """[Evaluator] Check evaluate with flush_preprocess."""
    with open(os.devnull, 'w') as f, redirect_stderr(f):
        # Silences the training output, but lets warnings and errors through.
        evals1.evaluate(estimators, parameters, X, y, 3, reset_preprocess=True)

    check_scores(evals1)


def test_pickling_evals():
    """[Evaluator] Test Pickling evals."""
    with open(os.devnull, 'w') as f, redirect_stderr(f):
        # Silences the training output, but lets warnings and errors through.
        evals2.evaluate(estimators, parameters, X, y, 3, flush_preprocess=True)
    pickle_save(evals2, 'evals_test_pickle1')
    pickled_eval1 = pickle_load('evals_test_pickle1')

    assert not hasattr(evals2, 'dout')
    assert not hasattr(pickled_eval1, 'dout')

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        evals2.evaluate(estimators, parameters, X, y, 3)
    pickle_save(evals2, 'evals_test_pickle2')
    pickled_eval2 = pickle_load('evals_test_pickle2')

    subprocess.check_call(['rm', 'evals_test_pickle1.pkl'])
    subprocess.check_call(['rm', 'evals_test_pickle2.pkl'])

    check_scores(pickled_eval1)
    check_scores(pickled_eval2)


def test_raise_error():
    """[Evaluator] Check that a FitFailedError is thrown."""
    evals1.error_score = None
    try:
        with open(os.devnull, 'w') as f, redirect_stderr(f):
            evals1.evaluate(estimators, parameters_exception, X, y, 3)
    except Exception as e:
        assert issubclass(type(e), FitFailedError)


def test_raise_warning():
    """[Evaluator] Check for a FitFailedWarning is thrown with error score."""
    # Set an error score
    evals1.error_score = -99

    # Suppress printed messages
    evals1.verbose = False

    # We need single threading for catch_warnings to record messages
    evals1.n_jobs_estimators = 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        evals1.evaluate(estimators, parameters_exception, X, y, 3)

    # Check warning
    assert len(w) == (evals1.n_iter * evals1.cv.n_splits *
                      len(evals1.preprocessing))
    assert all([issubclass(m.category, FitFailedWarning) for m in w])
    assert str(w[0].message) == \
        ("Could not fit estimator [rf]. Score set to -99. Details:\n"
         "ValueError('min_samples_leaf must be at least 1 or in (0, 0.5], "
         "got 1.58057518888',)")

    # Check Scores
    for i in [1, 2]:
        assert str(evals1.summary_.iloc[-i][0])[:3] == '-99'
