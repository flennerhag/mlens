""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Manual | Blend | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'blend', False, False)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Manual | Blend | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'blend', False, False)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Manual | Blend | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'blend', False, False)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Manual | Blend | No Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'blend', False, True)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Manual | Blend | No Proba | Prep] test predict"""
    args = get_layer('predict', 'manual', 'blend', False, True)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Manual | Blend | No Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'blend', False, True)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Manual | Blend | Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'blend', True, False)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Manual | Blend | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'blend', True, False)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Manual | Blend | Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'blend', True, False)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Manual | Blend | Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'blend', True, True)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Manual | Blend | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'blend', True, True)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Manual | Blend | Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'blend', True, True)
    run_layer(*args)
