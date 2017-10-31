""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Manual | Full | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'full', False, False)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Manual | Full | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'full', False, False)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Manual | Full | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'full', False, False)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Manual | Full | No Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'full', False, True)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Manual | Full | No Proba | Prep] test predict"""
    args = get_layer('predict', 'manual', 'full', False, True)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Manual | Full | No Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'full', False, True)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Manual | Full | Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'full', True, False)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Manual | Full | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'full', True, False)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Manual | Full | Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'full', True, False)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Manual | Full | Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'full', True, True)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Manual | Full | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'full', True, True)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Manual | Full | Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'full', True, True)
    run_layer(*args)
