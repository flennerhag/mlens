"""

Testing ground for parallel backend

"""
import numpy as np

from mlens.base import FoldIndex
from mlens.externals.sklearn.base import clone
from mlens.utils.dummy import destroy_temp_dir, _layer_est, _store_X_y, \
    data, \
    get_layers, get_path, ground_truth


PROFILE = False

LEN = 6
WIDTH = 2
FOLDS = 3
MOD, r = divmod(LEN, FOLDS)
assert r == 0

LAYER, LAYER_CONTAINER, LCM = get_layers('stack', True, n_splits=FOLDS)

X, _ = data((LEN, WIDTH), MOD)
y = np.arange(6) // 3

indexer = FoldIndex(FOLDS, X=X)
LAYER.indexer.fit(X)

(F, wf), (P, wp) = ground_truth(X, y, indexer, 'predict_proba', 2,
                                verbose=False)


def test_layer_fit():
    """[Layer] Stack proba: 'fit' method runs correctly."""

    layer = clone(LAYER)

    # Check predictions against ground truth
    preds = _layer_est(layer, 'fit', train=X, label=y,
                       n_jobs=-1, rem=True)

    np.testing.assert_array_equal(preds, F)

    # Check coefficients
    d = layer.estimators_
    ests = [(c, tup) for c, tup in d if c not in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wf


def test_layer_mmaps():
    """[Layer] Stack proba: no ests point to mmaps."""

    layer = clone(LAYER)

    # Check predictions against ground truth
    preds = _layer_est(layer, 'fit', train=X, label=y,
                       n_jobs=-1, rem=True)

    assert preds.__class__.__name__ == 'ndarray'

    for i in layer.estimators_:
        assert i[1][1].coef_.__class__.__name__ == 'ndarray'


def test_lc_fit():
    """[LayerContainer] Stack proba: 'fit' method runs correctly."""

    lc = clone(LAYER_CONTAINER)
    out = lc.fit(X, y, return_preds=True)

    # Test preds
    np.testing.assert_array_equal(F, out[-1])

    # Test coefs
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d if c not in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wf


def test_lc_mmaps():
    """[LayerContainer] Stack proba: no ests point to mmaps from."""
    lc = clone(LAYER_CONTAINER)
    out = lc.fit(X, y, return_preds=True)

    assert out[-1].__class__.__name__ == 'ndarray'

    for i in lc.layers['layer-1'].estimators_:
        assert i[1][1].coef_.__class__.__name__ == 'ndarray'


def test_layer_predict():
    """[Layer] Stack proba: 'predict' method runs correctly."""

    layer = clone(LAYER)

    # Check predictions against ground truth
    _ = _layer_est(layer, 'fit', train=X, label=y,
                   n_jobs=-1, rem=True)
    preds = _layer_est(layer, 'predict', train=X, label=y,
                       n_jobs=-1, rem=True, args=['X', 'P'])

    # Check predictions against GT
    np.testing.assert_array_equal(preds, P)

    # Check weights
    d = layer.estimators_
    ests = [(c, tup) for c, tup in d if c in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wp


def test_layer_transform():
    """[Layer] Stack proba: 'transform' method runs correctly."""

    layer = clone(LAYER)

    # Check predictions against ground truth
    _ = _layer_est(layer, 'fit', train=X, label=y,
                   n_jobs=-1, rem=True)
    preds = _layer_est(layer, 'transform', train=X, label=y,
                       n_jobs=-1, rem=True, args=['X', 'P'])

    # Check predictions against GT
    np.testing.assert_array_equal(preds, F)


def test_lc_predict():
    """[LayerContainer] Stack proba: 'predict' method runs correctly."""

    lc = clone(LAYER_CONTAINER)
    lc.fit(X, y)

    pred = lc.predict(X)

    # Test preds
    np.testing.assert_array_equal(P, pred)

    # Test coefs
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d if c in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    assert w == wp


def test_lc_transform():
    """[LayerContainer] Stack proba: 'transform' method runs correctly."""

    lc = clone(LAYER_CONTAINER)
    lc.fit(X, y)

    pred = lc.transform(X)

    np.testing.assert_array_equal(pred, F)


def test_lc_fit_from_file():
    """[LayerContainer] Stack proba: test fit from file path."""
    path = get_path()
    X_path, y_path = _store_X_y(path, X, y)

    lc = clone(LAYER_CONTAINER)
    out = lc.fit(X_path, y_path, return_preds=True)

    np.testing.assert_array_equal(F, out[-1])
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d if c not in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    destroy_temp_dir(path)

    assert w == wf


def test_lc_mmap_from_file():
    """[LayerContainer] Stack proba: no ests point to mmaps when fit from file."""
    path = get_path()
    X_path, y_path = _store_X_y(path, X, y)

    lc = clone(LAYER_CONTAINER)
    out = lc.fit(X_path, y_path, return_preds=True)

    destroy_temp_dir(path)

    assert out[-1].__class__.__name__ == 'ndarray'
    for e in lc.layers['layer-1'].estimators_:
        assert e[1][1].coef_.__class__.__name__ == 'ndarray'


def test_lc_predict_file():
    """[LayerContainer] Stack proba: test predict from file path."""
    path = get_path()
    X_path, y_path = _store_X_y(path, X, y)

    lc = clone(LAYER_CONTAINER)
    lc.fit(X_path, y_path)
    pred = lc.predict(X_path)

    np.testing.assert_array_equal(P, pred)
    d = lc.layers['layer-1'].estimators_
    ests = [(case, tup) for case, tup in d if case in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    destroy_temp_dir(path)

    assert w == wp


# Test preprocessors
#for tup in LAYER.preprocessing_:
#    if tup[1]:
#        print(tup[1][0][1].transform(X))

