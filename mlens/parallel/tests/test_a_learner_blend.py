""""ML-ENSEMBLE

Testing suite for Learner and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_learner, run_learner


def test_fit():
    """[Parallel | Learner | Blend | No Proba | No Prep] test fit"""
    args = get_learner('fit', 'blend', False, False)
    run_learner(*args)


def test_predict():
    """[Parallel | Learner | Blend | No Proba | No Prep] test predict"""
    args = get_learner('predict', 'blend', False, False)
    run_learner(*args)


def test_transform():
    """[Parallel | Learner | Blend | No Proba | No Prep] test transform"""
    args = get_learner('transform', 'blend', False, False)
    run_learner(*args)


def test_fit_prep():
    """[Parallel | Learner | Blend | No Proba | Prep] test fit"""
    args = get_learner('fit', 'blend', False, True)
    run_learner(*args)


def test_predict_prep():
    """[Parallel | Learner | Blend | No Proba | Prep] test predict"""
    args = get_learner('predict', 'blend', False, True)
    run_learner(*args)


def test_transform_prep():
    """[Parallel | Learner | Blend | No Proba | Prep] test transform"""
    args = get_learner('transform', 'blend', False, True)
    run_learner(*args)


def test_fit_proba():
    """[Parallel | Learner | Blend | Proba | No Prep] test fit"""
    args = get_learner('fit', 'blend', True, False)
    run_learner(*args)


def test_predict_proba():
    """[Parallel | Learner | Blend | Proba | No Prep] test predict"""
    args = get_learner('predict', 'blend', True, False)
    run_learner(*args)


def test_transform_proba():
    """[Parallel | Learner | Blend | Proba | No Prep] test transform"""
    args = get_learner('transform', 'blend', True, False)
    run_learner(*args)


def test_fit_prep_proba():
    """[Parallel | Learner | Blend | Proba | Prep] test fit"""
    args = get_learner('fit', 'blend', True, True)
    run_learner(*args)


def test_predict_prep_proba():
    """[Parallel | Learner | Blend | Proba | No Prep] test predict"""
    args = get_learner('predict', 'blend', True, True)
    run_learner(*args)


def test_transform_prep_proba():
    """[Parallel | Learner | Blend | Proba | Prep] test transform"""
    args = get_learner('transform', 'blend', True, True)
    run_learner(*args)


args = get_learner('fit', 'blend', False, False)
run_learner(*args)
