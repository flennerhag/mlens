"""ML-Ensemble

Test of the fit, predict and transform wrappers on the learners
"""
import numpy as np
from mlens.testing import Data, EstimatorContainer
from mlens.parallel import Group, Learner, Pipeline, run as _run
from mlens.utils.dummy import OLS, Scale
from mlens.externals.sklearn.base import clone


try:
    from sklearn.utils.estimator_checks import check_estimator
    SKLEARN = True
except ImportError:
    check_estimator = None
    SKLEARN = False


est = EstimatorContainer()


def scorer(p, y): return np.mean(p - y)


data = Data('stack', False, True, True)
X, y = data.get_data((25, 4), 3)
(F, wf), (P, wp) = data.ground_truth(X, y,)


if SKLEARN:
    def test_pipeline():
        """[Parallel | Pipeline] Test estimator"""
        check_estimator(Pipeline(Scale()))


def test_fit():
    """[Parallel | Learner] Testing fit flags"""
    lr = Learner(OLS(), indexer=data.indexer, name='lr')
    assert not lr.__fitted__
    _run(lr, 'fit', X, y)
    assert lr.__fitted__


def test_collect():
    """[Parallel | Learner] Testing multiple collections"""
    lr = Learner(OLS(), indexer=data.indexer, name='lr')
    lr.__no_output__ = False
    a = _run(lr, 'fit', X, y, return_preds=True)
    b = _run(lr, 'fit', X, y, refit=False, return_preds=True)
    c = _run(lr, 'transform', X, y, return_preds=True)
    d = _run(lr, 'transform', X, y)

    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(a, d)


def test_clone():
    """[Parallel | Learner] Testing cloning"""
    lr = Learner(OLS(), indexer=data.indexer, name='lr')
    l = clone(lr)
    l.set_indexer(data.indexer)

    assert not l.__fitted__

    F = _run(l, 'fit', X, y, return_preds=True)
    H = _run(lr, 'fit', X, y, return_preds=True, refit=False)
    np.testing.assert_array_equal(F, H)


def test_data():
    """[Parallel | Learner] Test data attribute"""
    lr = Learner(OLS(), indexer=data.indexer, name='lr')
    lr.scorer = scorer

    _run(lr, 'fit', X, y, return_preds=True)

    assert lr.raw_data
    assert isinstance(lr.raw_data, list)
    assert isinstance(lr.data, dict)
    assert lr.data.__repr__()
    assert 'score' in lr.data.__repr__()


def run(cls, job, eval=True):
    """Run a test"""
    if job == 'fit':
        lr, _ = est.get_learner(cls, True, False)
        lr.dtype = np.float64
    else:
        lr = run(cls, 'fit', False)

    data = Data(cls, True, False, True)
    X, y = data.get_data((25, 4), 3)

    if job in ['fit', 'transform']:
        (F, wf), _ = data.ground_truth(X, y, data.indexer.partitions)
    else:
        _, (F, wf) = data.ground_truth(X, y, data.indexer.partitions)

    args = {'fit': [X, y], 'transform': [X], 'predict': [X]}[job]

    P = _run(lr, job, *args, return_preds=True)
    if not eval:
        return lr

    np.testing.assert_array_equal(P, F)

    if job in ['fit', 'transform']:
        lrs = lr.sublearners
    else:
        lrs = lr.learner

    w = [obj.estimator.coef_ for obj in lrs]
    np.testing.assert_array_equal(w, wf)


def test_transformer():
    """[Parallel | Transform] test run transformer as estimator"""
    _, tr = est.get_learner('stack', True, True)

    data = Data('stack', True, True, True)
    X, y = data.get_data((25, 4), 3)

    F = _run(tr, 'fit', X, y, return_preds=True)
    H = _run(tr, 'transform', X, y, return_preds=True)
    P = _run(tr, 'predict', X, y, return_preds=True)
    Z, _ = tr.estimator.fit_transform(X)
    G = _run(tr, 'fit', X, y, return_preds=True, refit=False)

    np.testing.assert_array_equal(H, F)
    np.testing.assert_array_equal(Z, P)
    np.testing.assert_array_equal(G, F)


def test_fit_blend():
    """[Parallel | Learner | Blend | Wrapper] test fit"""
    run('blend', 'fit')


def test_pred_blend():
    """[Parallel | Learner | Blend | Wrapper] test predict"""
    run('blend', 'predict')


def test_tr_blend():
    """[Parallel | Learner | Blend | Wrapper] test transform"""
    run('blend', 'transform')


def test_fit_subsemble():
    """[Parallel | Learner | Subsemble | Wrapper] test fit"""
    run('subsemble', 'fit')


def test_pred_subsemble():
    """[Parallel | Learner | Subsemble | Wrapper] test predict"""
    run('subsemble', 'predict')


def test_tr_subsemble():
    """[Parallel | Learner | Subsemble | Wrapper] test transform"""
    run('subsemble', 'transform')


def test_run_fit():
    """[Parallel | Wrapper] test fit with auxiliary"""
    lr, tr = EstimatorContainer().get_learner('stack', False, True)
    group = Group(learners=lr, transformers=tr, dtype=np.float64)

    A = _run(group, 'fit', X, y, return_preds=True)
    np.testing.assert_array_equal(A, F)


def test_run_transform():
    """[Parallel | Wrapper] test transform with auxiliary"""
    lr, tr = EstimatorContainer().get_learner('stack', False, True)
    group = Group(learners=lr, transformers=tr, dtype=np.float64)

    _run(group, 'fit', X, y)
    A = _run(group, 'transform', X)
    np.testing.assert_array_equal(A, F)


def test_run_predict():
    """[Parallel | Wrapper] test predict with auxiliary"""
    lr, tr = EstimatorContainer().get_learner('stack', False, True)
    group = Group(learners=lr, transformers=tr, dtype=np.float64)

    _run(group, 'fit', X, y)
    A = _run(group, 'predict', X)
    np.testing.assert_array_equal(A, P)
