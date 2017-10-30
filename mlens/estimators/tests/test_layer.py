"""ML-ENSEMBLE

Test classes.
"""
import numpy as np
from mlens.index import FoldIndex
from mlens.testing import Data
from mlens.testing.dummy import ESTIMATORS, PREPROCESSING
from mlens.utils.dummy import OLS, Scale
from mlens.utils.exceptions import NotFittedError
from mlens.parallel import make_group
from mlens.estimators import LearnerEstimator, TransformerEstimator, LayerEnsemble
from mlens.externals.sklearn.base import clone

try:
    from sklearn.utils.estimator_checks import check_estimator
    run_sklearn = True
except ImportError:
    check_estimator = None
    run_sklearn = False

data = Data('stack', False, True)
X, y = data.get_data((25, 4), 3)
(F, wf), (P, wp) = data.ground_truth(X, y)

Est = LayerEnsemble
est = LayerEnsemble(make_group(FoldIndex(), ESTIMATORS, PREPROCESSING),
                    dtype=np.float64)


class Tmp(Est):

    """Temporary class

    Wrapper to get full estimator on no-args instantiation. For compatibility
    with older Scikit-learn versions.
    """

    def __init__(self):
        args = {LearnerEstimator: (OLS(), FoldIndex()),
                LayerEnsemble: (make_group(
                    FoldIndex(), ESTIMATORS, PREPROCESSING),),
                TransformerEstimator: (Scale(), FoldIndex())}[Est]
        super(Tmp, self).__init__(*args)


# These are simple run tests to ensure parallel wrapper register backend.
# See parallel for more rigorous tests

def test_layer_fit():
    """[Module | LayerEstimator] test fit"""
    out = est.fit(X, y)
    assert out is est

    p = est.fit_transform(X, y, refit=False)
    np.testing.assert_array_equal(p, F)


def test_layer_transform():
    """[Module | LayerEnsemble] test transform"""
    p = est.transform(X)
    np.testing.assert_array_equal(p, F)


def test_layer_predict():
    """[Module | LayerEnsemble] test predict"""
    p = est.predict(X)
    np.testing.assert_array_equal(p, P)


def test_layer_clone():
    """[Module | LayerEnsemble] test clone"""
    cl = clone(est)
    p = cl.fit_transform(X, y)
    np.testing.assert_array_equal(p, F)


def test_layer_params_estimator():
    """[Module | LayerEnsemble] test set params on estimator"""
    est.fit(X, y)

    # Just a check that this works
    out = est.get_params()
    assert isinstance(out, dict)

    est.set_params(**{'offs-1__estimator__offset': 10})
    np.testing.assert_raises(NotFittedError, est.predict, X)


def test_layer_params_indexer():
    """[Module | LayerEnsemble] test set params on indexer"""
    est.fit(X, y)

    est.set_params(**{'null-1__indexer__folds': 3})
    np.testing.assert_raises(NotFittedError, est.predict, X)


def test_layer_attr():
    """[Module | LayerEnsemble] test setting attribute"""
    est.propagate_features = [0]
    assert not est.__fitted__

    # If this fails, it is trying to propagate feature but predict_out is None!
    est.fit(X, y)


if run_sklearn:
    def test_layer():
        """[Module | LayerEnsemble] test pass estimator checks"""
        check_estimator(Tmp)
