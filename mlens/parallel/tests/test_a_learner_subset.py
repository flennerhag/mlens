""""ML-ENSEMBLE

Testing suite for Learner and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_learner, run_learner


def test_fit():
    """[Parallel | Learner | Subset | No Proba | No Prep] test fit"""
    args = get_learner('fit', 'subsemble', False, False)
    run_learner(*args)


def test_predict():
    """[Parallel | Learner | Subset | No Proba | No Prep] test predict"""
    args = get_learner('predict', 'subsemble', False, False)
    run_learner(*args)


def test_transform():
    """[Parallel | Learner | Subset | No Proba | No Prep] test transform"""
    args = get_learner('transform', 'subsemble', False, False)
    run_learner(*args)


def test_fit_prep():
    """[Parallel | Learner | Subset | No Proba | Prep] test fit"""
    args = get_learner('fit', 'subsemble', False, True)
    run_learner(*args)


def test_predict_prep():
    """[Parallel | Learner | Subset | No Proba | Prep] test predict"""
    args = get_learner('predict', 'subsemble', False, True)
    run_learner(*args)


def test_transform_prep():
    """[Parallel | Learner | Subset | No Proba | Prep] test transform"""
    args = get_learner('transform', 'subsemble', False, True)
    run_learner(*args)


def test_fit_proba():
    """[Parallel | Learner | Subset | Proba | No Prep] test fit"""
    args = get_learner('fit', 'subsemble', True, False)
    run_learner(*args)


def test_predict_proba():
    """[Parallel | Learner | Subset | Proba | No Prep] test predict"""
    args = get_learner('predict', 'subsemble', True, False)
    run_learner(*args)


def test_transform_proba():
    """[Parallel | Learner | Subset | Proba | No Prep] test transform"""
    args = get_learner('transform', 'subsemble', True, False)
    run_learner(*args)


def test_fit_prep_proba():
    """[Parallel | Learner | Subset | Proba | Prep] test fit"""
    args = get_learner('fit', 'subsemble', True, True)
    run_learner(*args)


def test_predict_prep_proba():
    """[Parallel | Learner | Subset | Proba | No Prep] test predict"""
    args = get_learner('predict', 'subsemble', True, True)
    run_learner(*args)


def test_transform_prep_proba():
    """[Parallel | Learner | Subset | Proba | Prep] test transform"""
    args = get_learner('transform', 'subsemble', True, True)
    run_learner(*args)
