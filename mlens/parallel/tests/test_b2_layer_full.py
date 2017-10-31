""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import Data, EstimatorContainer, get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Threading | Full | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'threading', 'full', False, False)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Threading | Full | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'full', False, False)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Threading | Full | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'threading', 'full', False, False)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Threading | Full | No Proba | Prep] test fit"""
    args = get_layer('fit', 'threading', 'full', False, True)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Threading | Full | No Proba | Prep] test predict"""
    args = get_layer('predict', 'threading', 'full', False, True)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Threading | Full | No Proba | Prep] test transform"""
    args = get_layer('transform', 'threading', 'full', False, True)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Threading | Full | Proba | No Prep] test fit"""
    args = get_layer('fit', 'threading', 'full', True, False)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Threading | Full | Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'full', True, False)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Threading | Full | Proba | No Prep] test transform"""
    args = get_layer('transform', 'threading', 'full', True, False)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Threading | Full | Proba | Prep] test fit"""
    args = get_layer('fit', 'threading', 'full', True, True)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Threading | Full | Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'full', True, True)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Threading | Full | Proba | Prep] test transform"""
    args = get_layer('transform', 'threading', 'full', True, True)
    run_layer(*args)

def test_fit_fp():
    """[Parallel | Layer | Threading | Full | No Proba | No Prep] test feature prop fit"""
    args = get_layer('fit', 'threading', 'full', False, False, feature_prop=2)
    run_layer(*args)


def test_predict_fp():
    """[Parallel | Layer | Threading | Full | No Proba | No Prep] test feature prop predict"""
    args = get_layer('predict', 'threading', 'full', False, False, feature_prop=2)
    run_layer(*args)


def test_transform_fp():
    """[Parallel | Layer | Threading | Full | No Proba | No Prep] test feature prop transform"""
    args = get_layer('transform', 'threading', 'full', False, False, feature_prop=2)
    run_layer(*args)
