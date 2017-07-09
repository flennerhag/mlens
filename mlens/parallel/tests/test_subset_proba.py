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
                               lc_transform,
                               lc_feature_prop)

PROBA = True
PROCESSING = True
LEN = 12
WIDTH = 2
FOLDS = 3
PARTITIONS = 2
MOD, r = divmod(LEN, FOLDS)
assert r == 0

lg = LayerGenerator()
data = Data('subset', PROBA, PROCESSING, PARTITIONS, FOLDS)

X, y = data.get_data((LEN, WIDTH), MOD)
(F, wf), (P, wp) = data.ground_truth(X, y, subsets=PARTITIONS)

layer = lg.get_layer('subset', PROBA, PROCESSING, PARTITIONS, FOLDS)
lc = lg.get_layer_container('subset', PROBA, PROCESSING, PARTITIONS, FOLDS)
lc_p = lg.get_layer_container('subset', PROBA, PROCESSING, PARTITIONS, FOLDS,
                              propagate_features=[1])

layer.indexer.fit(X)

cache = Cache(X, y, data)


def test_layer_fit():
    """[Parallel | Subset | Prep | Proba] test layer fit."""
    layer_fit(layer, cache, F, wf)


def test_layer_predict():
    """[Parallel | Subset | Prep | Proba] test layer predict."""
    layer_predict(layer, cache, P, wp)


def test_layer_transform():
    """[Parallel | Subset | Prep | Proba] test layer transform."""
    layer_transform(layer, cache, F)


def test_lc_fit():
    """[Parallel | Subset | Prep | Proba] test layer container fit."""
    lc_fit(lc, X, y, F, wf)


def test_lc_predict():
    """[Parallel | Subset | Prep | Proba] test layer container predict."""
    lc_predict(lc, X, P, wp)


def test_lc_transform():
    """[Parallel | Subset | Prep | Proba] test layer container transform."""
    lc_transform(lc, X, F)


def test_lc_file():
    """[Parallel | Subset | Prep | Proba] test layer container from file."""
    lc_from_file(lc, cache, X, y, F, wf, P, wp)


def test_lc_csv():
    """[Parallel | Subset | Prep | Proba] test layer container from csv."""
    lc_from_csv(lc, cache, X, y, F, wf, P, wp)


def test_lc_prop():
    """[Parallel | Subset | Prep | Proba] test layer container feature propagation."""
    lc_feature_prop(lc_p, X, y, F)


def test_close():
    """[Parallel | Subset | Prep | Proba] close cache."""
    cache.terminate()
