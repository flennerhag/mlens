""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'stack', False, False)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'stack', False, False)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'stack', False, False)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'stack', False, True)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'stack', False, True)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'stack', False, True)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Multiprocessing | Stack | Proba | No Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'stack', True, False)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Multiprocessing | Stack | Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'stack', True, False)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Multiprocessing | Stack | Proba | No Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'stack', True, False)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Multiprocessing | Stack | Proba | Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'stack', True, True)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Multiprocessing | Stack | Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'stack', True, True)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Multiprocessing | Stack | Proba | Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'stack', True, True)
    run_layer(*args)


def test_fit_fp():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | No Prep] test feature prop fit"""
    args = get_layer('fit', 'multiprocessing', 'stack', False, False, feature_prop=2)
    run_layer(*args)


def test_predict_fp():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | No Prep] test feature prop predict"""
    args = get_layer('predict', 'multiprocessing', 'stack', False, False, feature_prop=2)
    run_layer(*args)


def test_transform_fp():
    """[Parallel | Layer | Multiprocessing | Stack | No Proba | No Prep] test feature prop transform"""
    args = get_layer('transform', 'multiprocessing', 'stack', False, False, feature_prop=2)
    run_layer(*args)
