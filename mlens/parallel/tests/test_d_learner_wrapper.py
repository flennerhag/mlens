"""ML-Ensemble

Test of the fit, predict and transform wrappers on the learners
"""
import numpy as np
from mlens.testing import Data, EstimatorContainer

est = EstimatorContainer()


def run(cls, job, eval=True):
    """Run a test"""
    if job == 'fit':
        lr, tr = est.get_learner(cls, True, True)
        lr.preprocess = tr
        lr.dtype = np.float64
    else:
        lr = run(cls, 'fit', False)

    data = Data(cls, True, True, True)
    X, y = data.get_data((25, 4), 3)

    if job in ['fit', 'transform']:
        (F, wf), _ = data.ground_truth(X, y, data.indexer.partitions)
    else:
        _, (F, wf) = data.ground_truth(X, y, data.indexer.partitions)

    args = {'fit': (X, y), 'transform': (X,), 'predict': (X,)}[job]

    P = getattr(lr, job)(*args, return_preds=True)
    if not eval:
        return lr

    if job in ['fit', 'transform']:
        lrs = lr.sublearners_
    else:
        lrs = lr.learner_

    np.testing.assert_array_equal(P, F)
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
