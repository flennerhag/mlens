"""ML-ENSEMBLE

Test base functionality.
"""

import numpy as np
from mlens.externals.sklearn.base import clone
from mlens.testing.dummy import Data, EstimatorContainer

LEN = 12
WIDTH = 4
MOD = 2

data = Data('stack', False, True, folds=3)
X, y = data.get_data((LEN, WIDTH), MOD)

lg = EstimatorContainer()
lc = lg.get_sequential('stack', False, False)
layer = lg.get_layer('stack', False, False)


def test_clone():
    """[Ensemble | Sequential] Test cloning."""
    cloned = clone(lc)

    params = lc.get_params(deep=False)
    params_cloned = cloned.get_params(deep=False)

    for par, param in params.items():
        if par == 'layers':
            assert param is not params_cloned[par]
        else:
            assert param is params_cloned[par]


def test_set_params():
    """[Ensemble | Sequential] Test set_params on estimators."""
    lc.set_params(**{'layer-1.ols-3.estimator__offset': 4})
    lr = [l for l in lc.layers[0].learners][-1]
    assert lr.estimator.offset == 4


def test_set_params_layer():
    """[Ensemble | Layer] Test set_params on estimators."""
    layer.set_params(**{'ols-3.estimator__offset': 2})
    lr = [l for l in layer.learners][-1]
    assert lr.estimator.offset == 2
