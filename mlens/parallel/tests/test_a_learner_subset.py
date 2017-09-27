""""ML-ENSEMBLE

Testing suite for Learner and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_learner, run_learner


def test_fit():
    """[Parallel | Learner | Subset | No Proba | No Prep] test fit"""
    args = get_learner('fit', 'subset', False, False)
    run_learner(*args)


def test_predict():
    """[Parallel | Learner | Subset | No Proba | No Prep] test predict"""
    args = get_learner('predict', 'subset', False, False)
    run_learner(*args)


def test_transform():
    """[Parallel | Learner | Subset | No Proba | No Prep] test transform"""
    args = get_learner('transform', 'subset', False, False)
    run_learner(*args)


def test_fit_prep():
    """[Parallel | Learner | Subset | No Proba | Prep] test fit"""
    args = get_learner('fit', 'subset', False, True)
    run_learner(*args)


def test_predict_prep():
    """[Parallel | Learner | Subset | No Proba | Prep] test predict"""
    args = get_learner('predict', 'subset', False, True)
    run_learner(*args)


def test_transform_prep():
    """[Parallel | Learner | Subset | No Proba | Prep] test transform"""
    args = get_learner('transform', 'subset', False, True)
    run_learner(*args)


def test_fit_proba():
    """[Parallel | Learner | Subset | Proba | No Prep] test fit"""
    args = get_learner('fit', 'subset', True, False)
    run_learner(*args)


def test_predict_proba():
    """[Parallel | Learner | Subset | Proba | No Prep] test predict"""
    args = get_learner('predict', 'subset', True, False)
    run_learner(*args)


def test_transform_proba():
    """[Parallel | Learner | Subset | Proba | No Prep] test transform"""
    args = get_learner('transform', 'subset', True, False)
    run_learner(*args)


def test_fit_prep_proba():
    """[Parallel | Learner | Subset | Proba | Prep] test fit"""
    args = get_learner('fit', 'subset', True, True)
    run_learner(*args)


def test_predict_prep_proba():
    """[Parallel | Learner | Subset | Proba | No Prep] test predict"""
    args = get_learner('predict', 'subset', True, True)
    run_learner(*args)


def test_transform_prep_proba():
    """[Parallel | Learner | Subset | Proba | Prep] test transform"""
    args = get_learner('transform', 'subset', True, True)
    run_learner(*args)
