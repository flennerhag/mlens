"""ML-ENSEMBLE

Test base functionality.
"""

import numpy as np
from mlens.externals.sklearn.base import clone
from mlens.testing.dummy import Data, EstimatorContainer

LEN = 12
WIDTH = 4
MOD = 2

data = Data('stack', False, True, n_splits=3)
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
    lc.set_params(**{'layer-1__ols-3__estimator__offset': 4})
    assert lc.layers[0].estimators[-1][1].offset == 4


def test_set_params_layer():
    """[Ensemble | Layer] Test set_params on estimators."""
    layer.set_params(**{'ols-3__estimator__offset': 4})
    assert layer.estimators[-1][1].offset == 4
