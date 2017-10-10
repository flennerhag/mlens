"""ML-Ensemble

Test of the fit, predict and transform wrappers on the learners
"""
import numpy as np
from mlens.testing import Data, EstimatorContainer

est = EstimatorContainer()


def run(cls, job, eval=True):
    """Run a test"""
    layer = cls.startswith('layer')
    if layer:
        cls = cls.split('__')[1]

    if job == 'fit':
        if layer:
            lr = est.get_layer(cls, True, True)
        else:
            lr, tr = est.get_learner(cls, True, True)
            lr.auxiliary = tr
        lr.dtype = np.float64
    else:
        lr = run(cls if not layer else 'layer__%s' % cls, 'fit', False)

    data = Data(cls, True, True, not layer)
    X, y = data.get_data((25, 4), 3)

    if job in ['fit', 'transform']:
        (F, wf), _ = data.ground_truth(X, y, data.indexer.partitions)
    else:
        _, (F, wf) = data.ground_truth(X, y, data.indexer.partitions)

    args = {'fit': (X, y), 'transform': (X,), 'predict': (X,)}[job]

    P = getattr(lr, job)(*args, return_preds=True)
    if not eval:
        return lr

    np.testing.assert_array_equal(P, F)

    if not layer:
        if job in ['fit', 'transform']:
            lrs = lr.sublearners_
        else:
            lrs = lr.learner_

        w = [obj.estimator.coef_ for obj in lrs]
        np.testing.assert_array_equal(w, wf)


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


def test_fit_layer_subsemble():
    """[Parallel | Layer | Subsemble | Wrapper] test fit"""
    run('layer__subsemble', 'fit')


def test_pred_layer_subsemble():
    """[Parallel | Layer | Subsemble | Wrapper] test predict"""
    run('layer__subsemble', 'predict')


def test_tr_layer_subsemble():
    """[Parallel | Layer | Subsemble | Wrapper] test transform"""
    run('layer__subsemble', 'transform')


def test_transformer():
    """Test that the transformer runs"""
    _, tr = est.get_learner('stack', True, True)

    data = Data('stack', True, True, True)
    X, y = data.get_data((25, 4), 3)

    F = tr.fit(X, y, return_preds=True)
    H = tr.transform(X)
    P = tr.predict(X)
    Z = tr.estimator[0][1].fit(X).transform(X)
    G = tr.fit(X, y, return_preds=True)

    np.testing.assert_array_equal(H, F)
    np.testing.assert_array_equal(Z, P)
    np.testing.assert_array_equal(G, F)
