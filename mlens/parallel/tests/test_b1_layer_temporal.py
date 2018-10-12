""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Manual | Temporal | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Manual | Temporal | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Manual | Temporal | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Manual | Temporal | No Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Manual | Temporal | No Proba | Prep] test predict"""
    args = get_layer('predict', 'manual', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Manual | Temporal | No Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Manual | Temporal | Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Manual | Temporal | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Manual | Temporal | Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Manual | Temporal | Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Manual | Temporal | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Manual | Temporal | Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)
