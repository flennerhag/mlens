"""ML-ENSEMBLE

:author: Sebastian Flennerhag
"""

from __future__ import division, print_function, with_statement

import numpy as np
from pandas import DataFrame

from mlens.metrics import rmse
from mlens.utils.exceptions import FitFailedWarning
from mlens.parallel.fit_predict import _parallel_estimation, base_predict
from mlens.parallel.fit_predict import fit_estimators, cross_validate
from mlens.parallel._fit_predict_functions import (_fit_score, _fit_ests,
                                                   _fit_ests_folds)
from mlens.parallel._fit_predict_functions import _predict_folds
from mlens.parallel._fit_predict_functions import _predict, _construct_matrix

from sklearn.linear_model import Lasso

import warnings

# training data
np.random.seed(100)
X = np.random.random((100, 10))

# noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

# Change scales
X[:, 0] *= 10
X[:, 1] += 10
X[:, 2] *= 5
X[:, 3] *= 3
X[:, 4] /= 10

estimator = Lasso(alpha=0.001, random_state=100)
estimator_bad = Lasso(alpha='a', random_state=100)


def test_fit_estimator_func():
    """[Parallel] _fit_ests: test run."""
    tup = (y, (X, 'test'), ('ls', estimator))
    out = _fit_ests(tup)

    assert out[0] == 'test'
    assert out[1] == 'ls'
    assert hasattr(out[2], 'coef_')


def test_fit_estimator_func_exception_handling():
    """[Parallel] _fit_ests: exception handling."""
    tup = (y, (X, 'test'), ('ls', estimator_bad))

    with warnings.catch_warnings(record=True) as w:
        out = _fit_ests(tup)

    assert issubclass(w[0].category, FitFailedWarning)
    for key in out:
        assert key is None


def test_fit_and_predict_func():
    """[Parallel] _predict_folds: test run."""
    tup = (X, X, y, y, [i for i in range(100)], 'test'), ('ls', estimator)
    out = _predict_folds((True, True) + tup)

    assert out[0] == 'test-ls'
    assert str(out[-1][0])[:16] == '0.751494071538'


def test_fit_and_predict_func_exception_handling():
    """[Parallel] _predict_folds: exception handling."""
    tup = (X, X, y, y, [i for i in range(100)], 'test'), ('ls', estimator_bad)

    with warnings.catch_warnings(record=True) as w:
        out = _predict_folds((True, True) + tup)

    assert issubclass(w[0].category, FitFailedWarning)
    for entry in out:
        assert entry is None


def test_predict_func():
    """[Parallel] _predict: test run."""
    estimator.fit(X, y)
    tup = ((X, 'test'), ('ls', estimator))
    out = _predict((True,) + tup)
    assert out[0] == 'test-ls'
    assert str(out[-1][0])[:16] == '0.751494071538'


def test_predict_func_exception_handling():
    """[Parallel] _predict: exception handling."""
    tup = ((X, 'test'), ('ls', estimator_bad))

    with warnings.catch_warnings(record=True) as w:
        out = _predict((True,) + tup)

    assert issubclass(w[0].category, FitFailedWarning)
    for entry in out:
        assert entry is None


def test_construct_matrix_no_folds():
    """[Parallel] _construct_matrix: check build on unfolded input."""
    M = _construct_matrix([('a', X[:, 0]),
                           ('b', X[:, 1])], 100, ['a', 'b'], False)
    assert (M[:, 0] == X[:, 0]).all()
    assert (M[:, 1] == X[:, 1]).all()


def test_construct_matrix_folds():
    """[Parallel] _construct_matrix: check build on folded input."""
    d = [('a', [i for i in range(50)], X[range(50), 0]),
         ('a', [i for i in range(50, 100)], X[range(50, 100), 0]),
         ('b', [i for i in range(50)], X[range(50), 1]),
         ('b', [i for i in range(50, 100)], X[range(50, 100), 1])]
    M = _construct_matrix(d, 100, ['a', 'b'], True)
    assert (M == X[:, :2]).all()


def test_fit_score_func():
    """[Parallel] _fit_score: test run."""
    tup = (X[:50], X[50:], y[:50], y[50:], 'test')
    out = _fit_score(estimator, 'ls', {'alpha': 0.1}, rmse, tup, 1, np.nan)

    assert out[0] == 'ls-test'
    assert str(out[1]) == '-0.33038563036'
    assert str(out[2])[:16] == '-0.248531764447'
    assert out[4] == 2

    tup = (X, X, y, y)
    out = _fit_score(estimator, 'ls', {'alpha': 0.1}, rmse, tup, 1, np.nan)

    assert out[0] == 'ls'


def test_fit_score_func_exception_handling():
    """[Parallel] _fit_score: exception handling."""
    tup = (X[:50], X[50:], y[:50], y[50:], 'test')

    with warnings.catch_warnings(record=True) as w:
        out = _fit_score(estimator_bad, 'ls', {'alpha': 'b'}, rmse, tup, 1, -1)

    assert issubclass(w[0].category, FitFailedWarning)
    assert out[0] == 'ls-test'
    assert str(out[1]) == '-1'
    assert str(out[2]) == '-1'
    assert out[4] == 2


def test_parallel_estimation():
    """[Parallel] _parallel_estimation: test run."""
    tup = [(X, X, y, y, [i for i in range(100)], 'test')]

    out = _parallel_estimation(_predict_folds, tup,
                               {'test': [('ls', estimator)]},
                               optional_args=(True, True),
                               n_jobs=1, verbose=False)
    assert out[0][0] == 'test-ls'
    assert str(out[0][2][0])[:16] == '0.751494071538'


def test_base_predict():
    """[Parallel] _predict: exception handling."""
    data = [[X[:50], X[50:], y[:50], y[50:], range(50, 100), 'test'],
            [X[50:], X[:50], y[50:], y[:50], range(50), 'test']]

    with warnings.catch_warnings(record=True) as w:
        (M, ests) = base_predict(data, {'test': [('ls', estimator),
                                                 ('bad', estimator_bad)]},
                                 None,
                                 (True, True), True, 100,
                                 ['test-ls', 'test-bad'], True, n_jobs=1)

    # Check warnings
    assert len(w) == 2
    assert all([issubclass(m.category, FitFailedWarning) for m in w])

    # Check that bad estimator was dropped
    assert len(ests) == 1
    assert ests[0] == 'test-ls'
    assert (M.shape[0] == 100) & (M.shape[1] == 1)

    # Check that base predict carried through as if there was no bad estimator
    assert isinstance(M, DataFrame)
    assert M.columns[0] == 'test-ls'
    assert str(M.iloc[0, 0])[:16] == '0.807841612421'


def test_fit_estimators():
    """[Parallel] fit_estimators: test run."""
    tup = [(X, 'test')]

    out = fit_estimators(tup, {'test': [('ls', Lasso(alpha=0.001))]}, y,
                         None)

    assert isinstance(out, dict)
    assert isinstance(out['test'], list)
    assert out['test'][0][0] == 'ls'
    assert hasattr(out['test'][0][1], 'coef_')


def test_cross_validate():
    """[Parallel] cross_validate: test run."""
    tup = [[X[:50], X[50:], y[:50], y[50:], 'test']]
    out = cross_validate([('ls', Lasso())],
                         {'ls': [{'alpha': 0.31512612017405128},
                                 {'alpha': 0.82931980191796162}]},
                         tup, rmse, n_jobs=1)
    assert len(out) == 2
    assert all(out[i][0] == 'ls-test' for i in range(2))
    assert str(out[0][1]) == '-0.509895108976'
    assert str(out[0][2]) == '-0.442959652855'
    assert out[0][4] == 1
    assert str(out[1][1]) == '-0.514710787178'
    assert str(out[1][2]) == '-0.449490990214'
    assert out[1][4] == 2


def test_fit_ests_folds():
    """[Parallel] _fit_ests_folds: test run."""
    tup = [(X[:50], None, y[:50], None, None, 1), ('est', estimator)]
    out = _fit_ests_folds(tup)

    assert out[0] == 1
    assert out[1] == 'est'
    assert str(rmse(out[2], X[50:], y[50:])) == '-0.183772704162'


def test_fit_ests_folds_exception_handling():
    """[Parallel] _fit_ests_folds: exception handling."""
    tup = [(X[:50], None, y[:50], None, None, 1), ('est', estimator_bad)]

    with warnings.catch_warnings(record=True) as w:
        out = _fit_ests_folds(tup)

    assert issubclass(w[0].category, FitFailedWarning)
    for key in out:
        assert key is None
