"""

Testing ground for parallel backend

"""
from mlens.utils.dummy import LayerGenerator, Data, Cache
from mlens.utils.dummy import (layer_fit,
                               layer_predict,
                               layer_transform,
                               lc_fit,
                               lc_from_file,
                               lc_from_csv,
                               lc_predict,
                               lc_transform)

PROBA = False
PROCESSING = True
LEN = 6
WIDTH = 2
FOLDS = 3
MOD, r = divmod(LEN, FOLDS)
assert r == 0

lg = LayerGenerator()
data = Data('stack', PROBA, PROCESSING, FOLDS)

X, y = data.get_data((LEN, WIDTH), MOD)
(F, wf), (P, wp) = data.ground_truth(X, y)

layer = lg.get_layer('stack', PROBA, PROCESSING, FOLDS)
lc = lg.get_layer_container('stack', PROBA, PROCESSING, FOLDS)

layer.indexer.fit(X)

cache = Cache(X, y, data)


def test_layer_fit():
    """[Parallel | Stack | Prep] test layer fit."""
    layer_fit(layer, cache, F, wf)


def test_layer_predict():
    """[Parallel | Stack | Prep] test layer predict."""
    layer_predict(layer, cache, P, wp)


def test_layer_transform():
    """[Parallel | Stack | Prep] test layer transform."""
    layer_transform(layer, cache, F)


def test_lc_fit():
    """[Parallel | Stack | Prep] test layer container fit."""
    lc_fit(lc, X, y, F, wf)


def test_lc_predict():
    """[Parallel | Stack | Prep] test layer container predict."""
    lc_predict(lc, X, P, wp)


def test_lc_transform():
    """[Parallel | Stack | Prep] test layer container transform."""
    lc_transform(lc, X, F)


def test_lc_file():
    """[Parallel | Stack | Prep] test layer container input from file."""
    lc_from_file(lc, cache, X, y, F, wf, P, wp)


def test_lc_csv():
    """[Parallel | Stack | Prep] test layer container input from csv."""
    lc_from_csv(lc, cache, X, y, F, wf, P, wp)

def test_close():
    """[Parallel | Stack | Prep] close cache."""
    cache.terminate()
