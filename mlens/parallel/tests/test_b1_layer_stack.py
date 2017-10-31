""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Manual | Stack | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'stack', False, False)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Manual | Stack | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'stack', False, False)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Manual | Stack | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'stack', False, False)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Manual | Stack | No Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'stack', False, True)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Manual | Stack | No Proba | Prep] test predict"""
    args = get_layer('predict', 'manual', 'stack', False, True)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Manual | Stack | No Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'stack', False, True)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Manual | Stack | Proba | No Prep] test fit"""
    args = get_layer('fit', 'manual', 'stack', True, False)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Manual | Stack | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'stack', True, False)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Manual | Stack | Proba | No Prep] test transform"""
    args = get_layer('transform', 'manual', 'stack', True, False)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Manual | Stack | Proba | Prep] test fit"""
    args = get_layer('fit', 'manual', 'stack', True, True)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Manual | Stack | Proba | No Prep] test predict"""
    args = get_layer('predict', 'manual', 'stack', True, True)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Manual | Stack | Proba | Prep] test transform"""
    args = get_layer('transform', 'manual', 'stack', True, True)
    run_layer(*args)
