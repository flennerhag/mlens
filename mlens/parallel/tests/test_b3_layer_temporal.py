""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'temporal', False, False, window=2, step_size=3)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'temporal', False, True, window=2, step_size=3)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Multiprocessing | Temporal | Proba | No Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Multiprocessing | Temporal | Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Multiprocessing | Temporal | Proba | No Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'temporal', True, False, window=2, step_size=3)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Multiprocessing | Temporal | Proba | Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Multiprocessing | Temporal | Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Multiprocessing | Temporal | Proba | Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'temporal', True, True, window=2, step_size=3)
    run_layer(*args)

def test_fit_fp():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | No Prep] test feature prop fit"""
    args = get_layer('fit', 'multiprocessing', 'temporal', False, False, feature_prop=2, window=2, step_size=3)
    run_layer(*args)


def test_predict_fp():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | No Prep] test feature prop predict"""
    args = get_layer('predict', 'multiprocessing', 'temporal', False, False, feature_prop=2, window=2, step_size=3)
    run_layer(*args)


def test_transform_fp():
    """[Parallel | Layer | Multiprocessing | Temporal | No Proba | No Prep] test feature prop transform"""
    args = get_layer('transform', 'multiprocessing', 'temporal', False, False, feature_prop=2, window=2, step_size=3)
    run_layer(*args)
