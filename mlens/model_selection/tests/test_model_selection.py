"""ML-ENSEMBLE

Test model selection.
"""
import os
import numpy as np
from mlens.model_selection import Evaluator, benchmark
from mlens.metrics import mape, make_scorer
from mlens.utils.exceptions import FitFailedWarning
from mlens.utils.dummy import OLS, Scale
from mlens.testing import Data
from scipy.stats import randint

try:
    from contextlib import redirect_stdout, redirect_stderr
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stdout
    redirect_stderr = redirect_stdout

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


def test_params():
    """[Model Selection] Test raises on bad params."""
    evl = Evaluator(mape_scorer, verbose=2)

    np.testing.assert_raises(ValueError,
                             evl.fit, X, y,
                             estimators=[OLS()],
                             param_dicts={'bad.ols':
                                          {'offset': randint(1, 10)}},
                             preprocessing={'prep': [Scale()]})


def test_raises():
    """[Model Selection] Test raises on error."""

    evl = Evaluator(bad_scorer, verbose=1)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        np.testing.assert_raises(
            ValueError, evl.fit, X, y, estimators=[OLS()],
            param_dicts={'ols': {'offset': randint(1, 10)}}, n_iter=1)


def test_passes():
    """[Model Selection] Test sets error score on failed scoring."""

    evl = Evaluator(bad_scorer, error_score=0, n_jobs=1, verbose=5)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        evl = np.testing.assert_warns(FitFailedWarning,
                                      evl.fit, X, y,
                                      estimators=[OLS()],
                                      param_dicts={'ols':
                                                   {'offset': randint(1, 10)}},
                                      n_iter=1)
    assert evl.results['test_score-m']['ols'] == 0


def test_no_prep():
    """[Model Selection] Test run without preprocessing."""
    evl = Evaluator(mape_scorer, cv=5, shuffle=False,
                    random_state=100, verbose=12)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        evl.fit(X, y,
                estimators=[OLS()],
                param_dicts={'ols': {'offset': randint(1, 10)}},
                n_iter=3)

    np.testing.assert_approx_equal(
            evl.results['test_score-m']['ols'],
            -24.903229451043195)
    assert evl.results['params']['ols']['offset'] == 4


def test_w_prep_fit():
    """[Model Selection] Test run with preprocessing, single step."""
    evl = Evaluator(mape_scorer, cv=5, shuffle=False, random_state=100,
                    verbose=True)

    with open(os.devnull, 'w') as f, redirect_stdout(f):

        evl.fit(X, y,
                estimators=[OLS()],
                param_dicts={'ols': {'offset': randint(1, 10)}},
                preprocessing={'pr': [Scale()], 'no': []},
                n_iter=3)

    np.testing.assert_approx_equal(
            evl.results['test_score-m']['no.ols'],
            -24.903229451043195)

    np.testing.assert_approx_equal(
            evl.results['test_score-m']['pr.ols'],
            -26.510708862278072, 1)

    assert evl.results['params']['no.ols']['offset'] == 4
    assert evl.results['params']['pr.ols']['offset'] == 4


def test_w_prep_list_fit():
    """[Model Selection] Test run with preprocessing as list."""
    evl = Evaluator(
        mape_scorer, cv=5, shuffle=False, random_state=100, verbose=2)

    with open(os.devnull, 'w') as f, redirect_stdout(f):

        evl.fit(X, y,
                estimators=[OLS()],
                param_dicts={'ols': {'offset': randint(1, 10)}},
                preprocessing=[Scale()], n_iter=3)

    np.testing.assert_approx_equal(
            evl.results['test_score-m']['pr.ols'],
            -26.510708862278072)

    assert evl.results['params']['pr.ols']['offset'] == 4


def test_w_prep_set_params():
    """[Model Selection] Test run with preprocessing, sep param dists."""
    evl = Evaluator(mape_scorer, cv=5, shuffle=False, random_state=100,
                    verbose=2)

    params = {'no.ols': {'offset': randint(3, 6)},
              'pr.ols': {'offset': randint(1, 3)},
              }

    with open(os.devnull, 'w') as f, redirect_stdout(f):

        evl.fit(X, y,
                estimators={'pr': [OLS()], 'no': [OLS()]},
                param_dicts=params,
                preprocessing={'pr': [Scale()], 'no': []},
                n_iter=10)

    np.testing.assert_approx_equal(
            evl.results['test_score-m']['no.ols'],
            -18.684229451043198)

    np.testing.assert_approx_equal(
            evl.results['test_score-m']['pr.ols'],
            -7.2594502123869491)
    assert evl.results['params']['no.ols']['offset'] == 3
    assert evl.results['params']['pr.ols']['offset'] == 1


def test_bench_equality():
    """[Model Selection] Test benchmark correspondence with eval."""

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        evl = Evaluator(mape_scorer, cv=5)
        evl.fit(X, y, estimators={'pr': [OLS()], 'no': [OLS()]},
                param_dicts={}, preprocessing={'pr': [Scale()], 'no': []})

        out = benchmark(X, y, mape_scorer, 5, {'pr': [OLS()], 'no': [OLS()]},
                        {'pr': [Scale()], 'no': []}, None)

    np.testing.assert_approx_equal(out['test_score-m']['no.ols'],
                                   evl.results['test_score-m']['no.ols'])
