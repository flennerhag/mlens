"""ML-ENSEMBLE

Test model selection.
"""
import os
import numpy as np
from mlens.model_selection import Evaluator
from mlens.metrics import mape, make_scorer
from mlens.utils.exceptions import FitFailedWarning
from mlens.utils.dummy import Data, OLS, Scale
from scipy.stats import randint

try:
    from contextlib import redirect_stderr
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stderr

np.random.seed(100)

# Stack is nonsense here - we just need proba to be false
X, y = Data('stack', False, False).get_data((100, 2), 20)


def failed_score(p, y):
    """Bad scoring function to test exception handling."""
    raise ValueError("This fails.")

mape_scorer = make_scorer(mape, greater_is_better=False)
bad_scorer = make_scorer(failed_score)


def test_check():
    """[Model Selection] Test check of valid estimator."""
    np.testing.assert_raises(ValueError, Evaluator, mape)


def test_raises():
    """[Model Selection] Test raises on error."""

    evl = Evaluator(bad_scorer)

    np.testing.assert_raises(ValueError,
                             evl.fit, X, y, [OLS()],
                             {'ols': {'offset': randint(1, 10)}},
                             n_iter=1)


def test_passes():
    """[Model Selection] Test sets error score on failed scoring."""

    evl = Evaluator(bad_scorer, error_score=0, n_jobs=1)

    evl = np.testing.assert_warns(FitFailedWarning,
                                  evl.fit, X, y, [OLS()],
                                  {'ols': {'offset': randint(1, 10)}},
                                  n_iter=1)

    assert evl.summary['test_score_mean']['ols'] == 0


def test_no_prep():
    """[Model Selection] Test run without preprocessing."""
    evl = Evaluator(mape_scorer, verbose=True, cv=5, shuffle=False,
                    random_state=100)

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        evl.fit(X, y,
                estimators=[OLS()],
                param_dicts={'ols': {'offset': randint(1, 10)}},
                n_iter=3)

    np.testing.assert_approx_equal(
            evl.summary['test_score_mean']['ols'],
            -24.903229451043195)

    assert evl.summary['params']['ols']['offset'] == 4


def test_w_prep():
    """[Model Selection] Test run with preprocessing, double step."""
    evl = Evaluator(mape_scorer, cv=5, shuffle=False, random_state=100)

    # Preprocessing
    evl.preprocess(X, y, {'pr': [Scale()], 'no': []})

    # Fitting
    evl.evaluate(X, y,
                 estimators=[OLS()],
                 param_dicts={'ols': {'offset': randint(1, 10)}},
                 n_iter=3)

    np.testing.assert_approx_equal(
            evl.summary['test_score_mean'][('no', 'ols')],
            -24.903229451043195)

    np.testing.assert_approx_equal(
            evl.summary['test_score_mean'][('pr', 'ols')],
            -26.510708862278072, 1)

    assert evl.summary['params'][('no', 'ols')]['offset'] == 4
    assert evl.summary['params'][('pr', 'ols')]['offset'] == 4


def test_w_prep_fit():
    """[Model Selection] Test run with preprocessing, single step."""
    evl = Evaluator(mape_scorer, cv=5, shuffle=False, random_state=100)

    evl.fit(X, y,
            estimators=[OLS()],
            param_dicts={'ols': {'offset': randint(1, 10)}},
            preprocessing={'pr': [Scale()], 'no': []},
            n_iter=3)

    np.testing.assert_approx_equal(
            evl.summary['test_score_mean'][('no', 'ols')],
            -24.903229451043195)

    np.testing.assert_approx_equal(
            evl.summary['test_score_mean'][('pr', 'ols')],
            -26.510708862278072, 1)

    assert evl.summary['params'][('no', 'ols')]['offset'] == 4
    assert evl.summary['params'][('pr', 'ols')]['offset'] == 4


def test_w_prep_set_params():
    """[Model Selection] Test run with preprocessing, sep param dists."""
    evl = Evaluator(mape_scorer, cv=5, shuffle=False, random_state=100)

    params = {('no', 'ols'): {'offset': randint(3, 6)},
              ('pr', 'ols'): {'offset': randint(1, 3)},
              }

    # Fitting
    evl.fit(X, y,
            estimators={'pr': [OLS()], 'no': [OLS()]},
            param_dicts=params,
            preprocessing={'pr': [Scale()], 'no': []},
            n_iter=3)

    np.testing.assert_approx_equal(
            evl.summary['test_score_mean'][('no', 'ols')],
            -18.684229451043198)

    np.testing.assert_approx_equal(
            evl.summary['test_score_mean'][('pr', 'ols')],
            -7.2594502123869491, 1)

    assert evl.summary['params'][('no', 'ols')]['offset'] == 3
    assert evl.summary['params'][('pr', 'ols')]['offset'] == 1
