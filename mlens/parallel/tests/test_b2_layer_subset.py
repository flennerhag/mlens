""""ML-ENSEMBLE

Testing suite for Layer and Transformer
"""
from mlens.testing import get_layer, run_layer


def test_fit():
    """[Parallel | Layer | Threading | Subset | No Proba | No Prep] test fit"""
    args = get_layer('fit', 'threading', 'subsemble', False, False)
    run_layer(*args)


def test_predict():
    """[Parallel | Layer | Threading | Subset | No Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'subsemble', False, False)
    run_layer(*args)


def test_transform():
    """[Parallel | Layer | Threading | Subset | No Proba | No Prep] test transform"""
    args = get_layer('transform', 'threading', 'subsemble', False, False)
    run_layer(*args)


def test_fit_prep():
    """[Parallel | Layer | Threading | Subset | No Proba | Prep] test fit"""
    args = get_layer('fit', 'threading', 'subsemble', False, True)
    run_layer(*args)


def test_predict_prep():
    """[Parallel | Layer | Threading | Subset | No Proba | Prep] test predict"""
    args = get_layer('predict', 'threading', 'subsemble', False, True)
    run_layer(*args)


def test_transform_prep():
    """[Parallel | Layer | Threading | Subset | No Proba | Prep] test transform"""
    args = get_layer('transform', 'threading', 'subsemble', False, True)
    run_layer(*args)


def test_fit_proba():
    """[Parallel | Layer | Threading | Subset | Proba | No Prep] test fit"""
    args = get_layer('fit', 'threading', 'subsemble', True, False)
    run_layer(*args)


def test_predict_proba():
    """[Parallel | Layer | Threading | Subset | Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'subsemble', True, False)
    run_layer(*args)


def test_transform_proba():
    """[Parallel | Layer | Threading | Subset | Proba | No Prep] test transform"""
    args = get_layer('transform', 'threading', 'subsemble', True, False)
    run_layer(*args)


def test_fit_prep_proba():
    """[Parallel | Layer | Threading | Subset | Proba | Prep] test fit"""
    args = get_layer('fit', 'threading', 'subsemble', True, True)
    run_layer(*args)


def test_predict_prep_proba():
    """[Parallel | Layer | Threading | Subset | Proba | No Prep] test predict"""
    args = get_layer('predict', 'threading', 'subsemble', True, True)
    run_layer(*args)


def test_transform_prep_proba():
    """[Parallel | Layer | Threading | Subset | Proba | Prep] test transform"""
    args = get_layer('transform', 'threading', 'subsemble', True, True)
    run_layer(*args)


def test_fit_fp():
    """[Parallel | Layer | Threading | Subset | No Proba | No Prep] test feature prop fit"""
    args = get_layer('fit', 'threading', 'subsemble', False, False, feature_prop=2)
    run_layer(*args)


def test_predict_fp():
    """[Parallel | Layer | Threading | Subset | No Proba | No Prep] test feature prop predict"""
    args = get_layer('predict', 'threading', 'subsemble', False, False, feature_prop=2)
    run_layer(*args)


def test_transform_fp():
    """[Parallel | Layer | Threading | Subset | No Proba | No Prep] test feature prop transform"""
    args = get_layer('transform', 'threading', 'subsemble', False, False, feature_prop=2)
    run_layer(*args)
