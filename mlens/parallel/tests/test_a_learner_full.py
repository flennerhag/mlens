""""ML-ENSEMBLE

Testing suite for Learner and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_learner, run_learner


def test_predict():
    """[Parallel | Learner | Full | No Proba | No Prep] test fit and predict"""
    args = get_learner('predict', 'full', False, False)
    run_learner(*args)


def test_transform():
    """[Parallel | Learner | Full | No Proba | No Prep] test fit and transform"""
    args = get_learner('transform', 'full', False, False)
    run_learner(*args)


def test_predict_prep():
    """[Parallel | Learner | Full | No Proba | Prep] test fit and predict"""
    args = get_learner('predict', 'full', False, True)
    run_learner(*args)


def test_transform_prep():
    """[Parallel | Learner | Full | No Proba | Prep] test fit and transform"""
    args = get_learner('transform', 'full', False, True)
    run_learner(*args)


def test_predict_proba():
    """[Parallel | Learner | Full | Proba | No Prep] test fit and predict"""
    args = get_learner('predict', 'full', True, False)
    run_learner(*args)


def test_transform_proba():
    """[Parallel | Learner | Full | Proba | No Prep] test fit and transform"""
    args = get_learner('transform', 'full', True, False)
    run_learner(*args)


def test_predict_prep_proba():
    """[Parallel | Learner | Full | Proba | No Prep] test predict"""
    args = get_learner('predict', 'full', True, True)
    run_learner(*args)


def test_transform_prep_proba():
    """[Parallel | Learner | Full | Proba | Prep] test transform"""
    args = get_learner('transform', 'full', True, True)
    run_learner(*args)
