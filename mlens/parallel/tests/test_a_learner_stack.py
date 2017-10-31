""""ML-ENSEMBLE

Testing suite for Learner and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_learner, run_learner


def test_fit():
    """[Parallel | Learner | Stack | No Proba | No Prep] test fit"""
    args = get_learner('fit', 'stack', False, False)
    run_learner(*args)


def test_predict():
    """[Parallel | Learner | Stack | No Proba | No Prep] test predict"""
    args = get_learner('predict', 'stack', False, False)
    run_learner(*args)


def test_transform():
    """[Parallel | Learner | Stack | No Proba | No Prep] test transform"""
    args = get_learner('transform', 'stack', False, False)
    run_learner(*args)


def test_fit_prep():
    """[Parallel | Learner | Stack | No Proba | Prep] test fit"""
    args = get_learner('fit', 'stack', False, True)
    run_learner(*args)


def test_predict_prep():
    """[Parallel | Learner | Stack | No Proba | Prep] test predict"""
    args = get_learner('predict', 'stack', False, True)
    run_learner(*args)


def test_transform_prep():
    """[Parallel | Learner | Stack | No Proba | Prep] test transform"""
    args = get_learner('transform', 'stack', False, True)
    run_learner(*args)


def test_fit_proba():
    """[Parallel | Learner | Stack | Proba | No Prep] test fit"""
    args = get_learner('fit', 'stack', True, False)
    run_learner(*args)


def test_predict_proba():
    """[Parallel | Learner | Stack | Proba | No Prep] test predict"""
    args = get_learner('predict', 'stack', True, False)
    run_learner(*args)


def test_transform_proba():
    """[Parallel | Learner | Stack | Proba | No Prep] test transform"""
    args = get_learner('transform', 'stack', True, False)
    run_learner(*args)


def test_fit_prep_proba():
    """[Parallel | Learner | Stack | Proba | Prep] test fit"""
    args = get_learner('fit', 'stack', True, True)
    run_learner(*args)


def test_predict_prep_proba():
    """[Parallel | Learner | Stack | Proba | No Prep] test predict"""
    args = get_learner('predict', 'stack', True, True)
    run_learner(*args)


def test_transform_prep_proba():
    """[Parallel | Learner | Stack | Proba | Prep] test transform"""
    args = get_learner('transform', 'stack', True, True)
    run_learner(*args)
