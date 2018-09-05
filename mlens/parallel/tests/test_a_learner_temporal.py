""""ML-ENSEMBLE

Testing suite for Learner and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_learner, run_learner


def test_fit():
    """[Parallel | Learner | Temporal | No Proba | No Prep] test fit"""
    args = get_learner('fit', 'temporal', False, False, window=4, step_size=4)
    run_learner(*args)


def test_predict():
    """[Parallel | Learner | Temporal | No Proba | No Prep] test predict"""
    args = get_learner('predict', 'temporal', False, False, window=4, step_size=4)
    run_learner(*args)


def test_transform():
    """[Parallel | Learner | Temporal | No Proba | No Prep] test transform"""
    args = get_learner('transform', 'temporal', False, False, window=4, step_size=4)
    run_learner(*args)


def test_fit_prep():
    """[Parallel | Learner | Temporal | No Proba | Prep] test fit"""
    args = get_learner('fit', 'temporal', False, True, window=4, step_size=4)
    run_learner(*args)


def test_predict_prep():
    """[Parallel | Learner | Temporal | No Proba | Prep] test predict"""
    args = get_learner('predict', 'temporal', False, True, window=4, step_size=4)
    run_learner(*args)


def test_transform_prep():
    """[Parallel | Learner | Temporal | No Proba | Prep] test transform"""
    args = get_learner('transform', 'temporal', False, True, window=4, step_size=4)
    run_learner(*args)


def test_fit_proba():
    """[Parallel | Learner | Temporal | Proba | No Prep] test fit"""
    args = get_learner('fit', 'temporal', True, False, window=4, step_size=4)
    run_learner(*args)


def test_predict_proba():
    """[Parallel | Learner | Temporal | Proba | No Prep] test predict"""
    args = get_learner('predict', 'temporal', True, False, window=4, step_size=4)
    run_learner(*args)


def test_transform_proba():
    """[Parallel | Learner | Temporal | Proba | No Prep] test transform"""
    args = get_learner('transform', 'temporal', True, False, window=4, step_size=4)
    run_learner(*args)


def test_fit_prep_proba():
    """[Parallel | Learner | Temporal | Proba | Prep] test fit"""
    args = get_learner('fit', 'temporal', True, True, window=4, step_size=4)
    run_learner(*args)


def test_predict_prep_proba():
    """[Parallel | Learner | Temporal | Proba | No Prep] test predict"""
    args = get_learner('predict', 'temporal', True, True, window=4, step_size=4)
    run_learner(*args)


def test_transform_prep_proba():
    """[Parallel | Learner | Temporal | Proba | Prep] test transform"""
    args = get_learner('transform', 'temporal', True, True, window=4, step_size=4)
    run_learner(*args)
