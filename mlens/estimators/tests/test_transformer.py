"""ML-ENSEMBLE

Test classes.
"""
import numpy as np
from mlens.index import FoldIndex
from mlens.utils.dummy import OLS, Scale
from mlens.utils.exceptions import NotFittedError
from mlens.testing import Data
from mlens.estimators import LearnerEstimator, TransformerEstimator, LayerEnsemble
from mlens.externals.sklearn.base import clone

try:
    from sklearn.utils.estimator_checks import check_estimator
    run_sklearn = True
except ImportError:
    check_estimator = None
    run_sklearn = False

data = Data('stack', False, False)
X, y = data.get_data((25, 4), 3)


est = TransformerEstimator(Scale(), FoldIndex(), dtype=np.float64)

Est = TransformerEstimator


class Tmp(Est):

    """Temporary class

    Wrapper to get full estimator on no-args instantiation. For compatibility
    with older Scikit-learn versions.
    """

    def __init__(self):
        args = {LearnerEstimator: (OLS(), FoldIndex()),
                LayerEnsemble: (FoldIndex(), OLS()),
                TransformerEstimator: (Scale(), FoldIndex())}[Est]
        super(Tmp, self).__init__(*args)


# These are simple run tests to ensure parallel wrapper register backend.
# See parallel for more rigorous tests

def test_learner_fit():
    """[Module | TransformerEstimator] test fit"""
    out = est.fit(X, y)
    assert out is est
    p = est.fit_transform(X, y, refit=False)
    assert p.shape[0] == X.shape[0]
    assert p.shape[1] == X.shape[1]


def test_learner_transform():
    """[Module | TransformerEstimator] test transform"""
    p = est.transform(X)
    assert p.shape[0] == X.shape[0]
    assert p.shape[1] == X.shape[1]


def test_learner_predict():
    """[Module | TransformerEstimator] test predict"""
    p = est.predict(X)
    assert p.shape[0] == X.shape[0]
    assert p.shape[1] == X.shape[1]


def test_learner_clone():
    """[Module | TransformerEstimator] test clone"""
    cl = clone(est)
    cl.fit_transform(X, y)


def test_learner_params_estimator():
    """[Module | TransformerEstimator] test set params on estimator"""
    est.fit(X, y)

    # Just a check that this works
    out = est.get_params()
    assert isinstance(out, dict)

    est.set_params(preprocessing__copy=False)
    np.testing.assert_raises(NotFittedError, est.predict, X)


def test_learner_params_indexer():
    """[Module | TransformerEstimator] test set params on indexer"""
    est.fit(X, y)

    # Just a check that this works
    out = est.get_params()
    assert isinstance(out, dict)

    est.set_params(indexer__folds=3)
    np.testing.assert_raises(NotFittedError, est.predict, X)


def test_learner_attr():
    """[Module | TransformerEstimator] test setting attribute"""
    est.fit(X, y)
    est.indexer = FoldIndex(2)
    np.testing.assert_raises(NotFittedError, est.predict, X)


if run_sklearn:
    def test_learner():
        """[Module | TransformerEstimator] test pass estimator checks"""
        check_estimator(Tmp)
