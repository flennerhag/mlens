""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Manual | Subset | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'subset', False, False)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Manual | Subset | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'subset', False, False)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Manual | Subset | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'subset', False, False)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Manual | Subset | No Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'subset', False, True)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Manual | Subset | No Proba | Prep] test predict"""
    args = get_layer('predict', 'manual', 'subset', False, True)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Manual | Subset | No Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'subset', False, True)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Manual | Subset | Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'subset', True, False)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Manual | Subset | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'subset', True, False)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Manual | Subset | Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'subset', True, False)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Manual | Subset | Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'subset', True, True)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Manual | Subset | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'subset', True, True)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Manual | Subset | Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'subset', True, True)
    run_layer(*args)
