"""ML-ENSEMBLE

Test base functionality.
"""
from mlens.ensemble.base import BaseEnsemble
from mlens.externals.sklearn.base import clone
from mlens.testing.dummy import Data, EstimatorContainer

try:
    from sklearn.utils.estimator_checks import check_estimator
    SKLEARN = True
except ImportError:
    check_estimator = None
    SKLEARN = False

LEN = 12
WIDTH = 4
MOD = 2


class Tmp(BaseEnsemble):

    def __init__(
            self, shuffle=False, random_state=None, scorer=None, verbose=False,
            layers=None, array_check=2, model_selection=False, sample_size=20):
        super(Tmp, self).__init__(
            shuffle=shuffle, random_state=random_state, scorer=scorer,
            verbose=verbose, layers=layers, array_check=array_check,
            model_selection=model_selection, sample_size=sample_size)


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
        if par == 'stack':
            assert param is not params_cloned[par]
        else:
            assert param is params_cloned[par]


def test_set_params():
    """[Ensemble | Sequential] Test set_params on estimators."""
    lc.set_params(**{'layer-1__group__ols-3__estimator__offset': 4})
    lr = lc.stack[0].learners[-1]
    assert lr.estimator.offset == 4


def test_set_params_layer():
    """[Ensemble | Layer] Test set_params on estimators."""
    layer.set_params(**{'group__ols-3__estimator__offset': 2})
    lr = [l for l in layer.learners][-1]
    assert lr.estimator.offset == 2


if SKLEARN:
    def test_estimator_check():
        """[Ensemble | BaseEnsemble] Test valid scikit-learn estimator."""
        check_estimator(Tmp(layers=lc.stack))
