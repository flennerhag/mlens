"""

Testing ground for parallel backend

"""
import numpy as np

from mlens.base import BlendIndex
from mlens.externals.sklearn.base import clone
from mlens.utils.dummy import destroy_temp_dir, _layer_est, _store_X_y, \
    get_layers, get_path, ground_truth

np.random.seed(10)
X = np.arange(24).reshape(12, 2)

y = np.ones(12)
y[:3] = 1
y[3:8] = 2
y[8:12] = 3

np.random.shuffle(X)
np.random.shuffle(y)

LAYER, LAYER_CONTAINER, LCM = get_layers('blend', True, test_size=5)

indexer = BlendIndex(test_size=5, X=X)
LAYER.indexer.fit(X)

(F, wf), (P, wp) = ground_truth(X, y, indexer, 'predict_proba', 3,
                                verbose=False)


def test_layer_fit():
    """[Layer] Blend proba: 'fit' method runs correctly."""

    layer = clone(LAYER)

    # Check predictions against ground truth
    preds = _layer_est(layer, 'fit', train=X, label=y,
                       n_jobs=-1, rem=True)

    np.testing.assert_array_equal(preds, F)

    # Check coefficients
    d = layer.estimators_
    ests = [(c, tup) for c, tup in d if c in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wf


def test_layer_mmaps():
    """[Layer] Blend proba: no ests point to mmaps."""

    layer = clone(LAYER)

    # Check predictions against ground truth
    preds = _layer_est(layer, 'fit', train=X, label=y,
                       n_jobs=-1, rem=True)

    assert preds.__class__.__name__ == 'ndarray'

    for i in layer.estimators_:
        assert i[1][1].coef_.__class__.__name__ == 'ndarray'


def test_lc_fit():
    """[LayerContainer] Blend proba: 'fit' method runs correctly."""

    lc = clone(LAYER_CONTAINER)
    out = lc.fit(X, y, return_preds=True)

    # Test preds
    np.testing.assert_array_equal(F, out[-1])

    # Test coefs
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d if c in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wf


def test_lc_mmaps():
    """[LayerContainer] Blend proba: no ests point to mmaps from."""
    lc = clone(LAYER_CONTAINER)
    out = lc.fit(X, y, return_preds=True)

    assert out[-1].__class__.__name__ == 'ndarray'

    for i in lc.layers['layer-1'].estimators_:
        assert i[1][1].coef_.__class__.__name__ == 'ndarray'


def test_layer_predict():
    """[Layer] Blend proba: 'predict' method runs correctly."""

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


def test_lc_predict():
    """[LayerContainer] Blend proba: 'predict' method runs correctly."""

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


def test_lc_fit_from_file():
    """[LayerContainer] Blend proba: test fit from file path."""
    path = get_path()
    X_path, y_path = _store_X_y(path, X, y)

    lc = clone(LAYER_CONTAINER)
    out = lc.fit(X_path, y_path, return_preds=True)

    np.testing.assert_array_equal(F, out[-1])
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d if c in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    destroy_temp_dir(path)

    assert w == wf


def test_lc_mmap_from_file():
    """[LayerContainer] Blend proba: no ests point to mmaps when fit from file."""
    path = get_path()
    X_path, y_path = _store_X_y(path, X, y)

    lc = clone(LAYER_CONTAINER)
    out = lc.fit(X_path, y_path, return_preds=True)

    destroy_temp_dir(path)

    assert out[-1].__class__.__name__ == 'ndarray'
    for e in lc.layers['layer-1'].estimators_:
        assert e[1][1].coef_.__class__.__name__ == 'ndarray'


def test_lc_predict_file():
    """[LayerContainer] Blend proba: test predict from file path."""
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
