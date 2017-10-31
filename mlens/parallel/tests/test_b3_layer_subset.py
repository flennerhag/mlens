""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'subsemble', False, False)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'subsemble', False, False)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'subsemble', False, False)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'subsemble', False, True)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'subsemble', False, True)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'subsemble', False, True)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Multiprocessing | Subset | Proba | No Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'subsemble', True, False)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Multiprocessing | Subset | Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'subsemble', True, False)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Multiprocessing | Subset | Proba | No Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'subsemble', True, False)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Multiprocessing | Subset | Proba | Prep] test fit"""
    args = get_layer('fit', 'multiprocessing', 'subsemble', True, True)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Multiprocessing | Subset | Proba | No Prep] test predict"""
    args = get_layer('predict', 'multiprocessing', 'subsemble', True, True)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Multiprocessing | Subset | Proba | Prep] test transform"""
    args = get_layer('transform', 'multiprocessing', 'subsemble', True, True)
    run_layer(*args)

def test_fit_fp():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | No Prep] test feature prop fit"""
    args = get_layer('fit', 'multiprocessing', 'subsemble', False, False, feature_prop=2)
    run_layer(*args)


def test_predict_fp():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | No Prep] test feature prop predict"""
    args = get_layer('predict', 'multiprocessing', 'subsemble', False, False, feature_prop=2)
    run_layer(*args)


def test_transform_fp():
    """[Parallel | Layer | Multiprocessing | Subset | No Proba | No Prep] test feature prop transform"""
    args = get_layer('transform', 'multiprocessing', 'subsemble', False, False, feature_prop=2)
    run_layer(*args)
