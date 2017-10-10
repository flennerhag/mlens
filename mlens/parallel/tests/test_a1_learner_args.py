"""ML-ENSEMBLE

Test learner and transformer attributes
"""
import os
import numpy as np
from mlens.parallel import Learner, Transformer
from mlens.utils.dummy import OLS, Scale
from mlens.testing.dummy import Data
from mlens.externals.sklearn.base import clone
try:
    from contextlib import redirect_stdout
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stdout


data = Data('stack', True, True)


def scorer(p, y): return np.mean(p - y)

X, y = data.get_data((25, 4), 3)

tr = Transformer([('scale', Scale())], data.indexer, 'tr')
lr = Learner(OLS(), 'tr', data.indexer, 'lr', auxiliary=tr)


def run(l, attr, **kwargs):
    """Wrapper to graciously fail a fit or predict call"""
    if attr == 'fit':
        args = (X, y)
    else:
        args = (X,)
    try:
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            out = getattr(l, attr)(*args, **kwargs)
            return out
    except:
        raise AssertionError("Could not fit layer")


def test_fit():
    """[Parallel | Learner] Testing fit flags"""
    assert not lr.__fitted__
    assert tr.__no_output__
    run(lr, 'fit')
    assert lr.__fitted__
    assert tr.__no_output__


def test_collect():
    """[Parallel | Learner] Testing multiple collections"""
    a = run(lr, 'fit', return_preds=True)
    b = run(lr, 'fit', refit=False, return_preds=True)
    c = run(lr, 'transform')
    d = run(lr, 'fit', return_preds=True)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(a, d)


def test_clone():
    """[Parallel | Learner] Testing cloning"""
    l = clone(lr)

    assert not l.__fitted__
    F = run(l, 'fit', return_preds=True)
    H = run(lr, 'fit', return_preds=True, refit=False)
    np.testing.assert_array_equal(F, H)


def test_data():
    """[Parallel | Learner] Test data"""
    lr.scorer = scorer

    run(lr, 'fit')

    assert lr.raw_data
    assert isinstance(lr.raw_data, list)
    assert isinstance(lr.data, dict)
    assert lr.data.__repr__()
    assert 'score' in lr.data.__repr__()
