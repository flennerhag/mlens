"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Test layer push and pop ops.
"""
import os
import numpy as np
from mlens.index import INDEXERS
from mlens.testing.dummy import Data, ECM
from mlens.utils.dummy import Scale, LogisticRegression
from mlens.parallel import make_learners, Layer
from mlens.externals.sklearn.base import clone
try:
    from contextlib import redirect_stdout
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stdout


PREPROCESSING_1 = {'no1': [], 'sc1': [('scale', Scale())]}
ESTIMATORS_PROBA_1 = {'sc1': [('offs1', LogisticRegression(offset=2)),
                              ('null1', LogisticRegression())],
                      'no1': [('offs1', LogisticRegression(offset=2)),
                              ('null1', LogisticRegression())]}

PREPROCESSING_2 = {'no2': [], 'sc2': [('scale', Scale())]}
ESTIMATORS_PROBA_2 = {'sc2': [('offs2', LogisticRegression(offset=2)),
                             ('null2', LogisticRegression())],
                      'no2': [('offs2', LogisticRegression(offset=2)),
                             ('null2', LogisticRegression())]}


def scorer(p, y): return np.mean(p - y)


data = Data('stack', True, True)

X, y = data.get_data((10, 4), 3)

idx1 = INDEXERS['stack']()
g1 = make_learners(
    idx1, ESTIMATORS_PROBA_1, PREPROCESSING_1,
    learner_kwargs={'proba': True, 'verbose': True},
    transformer_kwargs={'verbose': True})

idx2 = INDEXERS['subsemble']()
g2 = make_learners(
    idx2, ESTIMATORS_PROBA_2, PREPROCESSING_2,
    learner_kwargs={'proba': False, 'verbose': True},
    transformer_kwargs={'verbose': True})

layer = Layer('layer')


def test_push_1():
    """[Parallel | Layer] Testing single push"""
    assert not layer.__initialized__

    layer.push(g1)

    assert layer.groups[0] is g1
    assert layer.__initialized__

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        layer.fit(X, y)


def test_push_2():
    """[Parallel | Layer] Testing double push"""

    assert layer.__fitted__
    layer.push(g2)
    assert not layer.__fitted__

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        a = layer.fit(X, y, refit=False, return_preds=True)

    assert layer.__fitted__

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        b = layer.fit(X, y, refit=False, return_preds=True)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        c = layer.transform(X)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        d = layer.fit(X, y, refit=True, return_preds=True)

    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(a, d)


def test_clone():
    """[Parallel | Layer] Testing cloning"""
    lyr = clone(layer)

    assert lyr.__initialized__
    assert not lyr.__fitted__

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        F = layer.fit(X, y, return_preds=True, refit=False)
        H = lyr.fit(X, y, return_preds=True)
    np.testing.assert_array_equal(F, H)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        F = layer.transform(X)
        H = lyr.transform(X)
    np.testing.assert_array_equal(F, H)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        F = layer.predict(X)
        H = lyr.predict(X)
    np.testing.assert_array_equal(F, H)


def test_data():
    """[Parallel | Learner] Test data"""
    idx = INDEXERS['subsemble']()
    lyr = Layer('layer-scorer').push(make_learners(idx, ECM, None))
    for lr in lyr.learners:
        lr.scorer = scorer

    lyr.fit(X, y)
    repr = lyr.data.__repr__()
    assert lyr.raw_data
    assert isinstance(lyr.raw_data, list)
    assert isinstance(lyr.data, dict)
    assert repr
    assert 'score' in repr


def test_pop():
    """[Parallel | Layer] Testing pop"""
    # Popping one group leaves the layer intact
    g = layer.pop(0)
    assert layer.__initialized__
    assert layer.__fitted__
    assert g1 is g

    # Popping both leaves if empty
    g = layer.pop(0)
    assert not layer.__initialized__
    assert not layer.__fitted__
    assert g2 is g

    # Pushing fitted groups makes the layer fitted
    layer.push(g1, g2)
    assert layer.__fitted__
