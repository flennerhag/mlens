""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Threading | Temporal | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'threading', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Threading | Temporal | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Threading | Temporal | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'threading', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Threading | Temporal | No Proba | Prep] test fit"""
    args = get_layer('fit', 'threading', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Threading | Temporal | No Proba | Prep] test predict"""
    args = get_layer('predict', 'threading', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Threading | Temporal | No Proba | Prep] test transform"""
    args = get_layer('transform', 'threading', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Threading | Temporal | Proba | No Prep] test fit"""
    args = get_layer('fit', 'threading', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Threading | Temporal | Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Threading | Temporal | Proba | No Prep] test transform"""
    args = get_layer('transform', 'threading', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Threading | Temporal | Proba | Prep] test fit"""
    args = get_layer('fit', 'threading', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Threading | Temporal | Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Threading | Temporal | Proba | Prep] test transform"""
    args = get_layer('transform', 'threading', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)

def test_fit_fp():
    """[Parallel | Layer | Threading | Temporal | No Proba | No Prep] test feature prop fit"""
    args = get_layer('fit', 'threading', 'temporal', False, False, feature_prop=2, window=2, step_size=3)
    run_layer(*args)


def test_predict_fp():
    """[Parallel | Layer | Threading | Temporal | No Proba | No Prep] test feature prop predict"""
    args = get_layer('predict', 'threading', 'temporal', False, False, feature_prop=2, window=2, step_size=3)
    run_layer(*args)


def test_transform_fp():
    """[Parallel | Layer | Threading | Temporal | No Proba | No Prep] test feature prop transform"""
    args = get_layer('transform', 'threading', 'temporal', False, False, feature_prop=2, window=2, step_size=3)
    run_layer(*args)
