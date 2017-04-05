"""ML-ENSEMBLE

Layer and Layer Container test functions.
"""
import numpy as np

LEN = 6
WIDTH = 2
FOLDS = 3
MOD, r = divmod(LEN, FOLDS)
assert r == 0


def layer_fit(layer, cache, F, wf):
    """Test the layer's fit method."""

    # Check predictions against ground truth
    preds = cache._layer_est(layer, 'fit', n_jobs=-1)
    np.testing.assert_array_equal(preds, F)

    # Check coefficients
    d = layer.estimators_
    if layer.cls != 'blend':
        d = d[layer.n_pred:]

    ests = [(c, tup) for c, tup in d]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wf

    assert preds.__class__.__name__ == 'ndarray'

    for i in layer.estimators_:
        assert i[1][1].coef_.__class__.__name__ == 'ndarray'


def layer_predict(layer, cache, P, wp):
    """Test the layer's predict method."""
    preds = cache._layer_est(layer, 'predict', n_jobs=-1, args=['X', 'P'])
    np.testing.assert_array_equal(preds, P)

    # Check weights
    d = layer.estimators_
    ests = [(c, tup) for c, tup in d[:layer.n_pred]]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wp


def layer_transform(layer, cache, F):
    """Test the layer's transform method."""

    # Check predictions against ground truth
    preds = cache._layer_est(layer, 'transform', n_jobs=-1, args=['X', 'P'])

    # Check predictions against GT
    np.testing.assert_array_equal(preds, F)


def lc_fit(lc, X, y, F, wf):
    """Test the layer containers fit method."""

    out = lc.fit(X, y, return_preds=True)

    # Test preds
    np.testing.assert_array_equal(F, out[-1])

    # Test coefs
    d = lc.layers['layer-1'].estimators_
    if lc.layers['layer-1'].cls != 'blend':
        d = d[lc.layers['layer-1'].n_pred:]

    ests = [(c, tup) for c, tup in d]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wf

    assert out[-1].__class__.__name__ == 'ndarray'

    for i in lc.layers['layer-1'].estimators_:
        assert i[1][1].coef_.__class__.__name__ == 'ndarray'


def lc_predict(lc, X, P, wp):
    """Test the layer containers predict method."""

    pred = lc.predict(X)

    # Test preds
    np.testing.assert_array_equal(P, pred)

    # Test coefs
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d[:lc.layers['layer-1'].n_pred]]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    assert w == wp


def lc_transform(lc, X, F):
    """Test the layer containers transform method."""

    pred = lc.transform(X)
    np.testing.assert_array_equal(pred, F)


def lc_from_file(lc, cache, X, y, F, wf, P, wp):
    """[LayerContainer] Stack: test fit from file path."""

    X_path, y_path = cache._store_X_y(X, y)

    # TEST FIT
    out = lc.fit(X_path, y_path, return_preds=True)

    np.testing.assert_array_equal(F, out[-1])
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d if c not in ['sc', 'no']]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    # TEST MMAP
    assert out[-1].__class__.__name__ == 'ndarray'
    for e in lc.layers['layer-1'].estimators_:
        assert e[1][1].coef_.__class__.__name__ == 'ndarray'

    # TEST PREDICT
    pred = lc.predict(X_path)

    np.testing.assert_array_equal(P, pred)
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d[:lc.layers['layer-1'].n_pred]]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    assert w == wp
