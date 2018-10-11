"""ML-ENSEMBLE

Test classes.
"""
import numpy as np
from mlens.index import FoldIndex, FullIndex
from mlens.utils.dummy import OLS, Scale
from mlens.utils.exceptions import NotFittedError, ParameterChangeWarning
from mlens.testing import Data
from mlens.estimators import LearnerEstimator, TransformerEstimator, LayerEnsemble
from mlens.externals.sklearn.base import clone

try:
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.utils.validation import check_X_y, check_array
    run_sklearn = True
except ImportError:
    check_estimator = None
    run_sklearn = False

data = Data('stack', False, False)
X, y = data.get_data((25, 4), 3)
(F, wf), (P, wp) = data.ground_truth(X, y)

est = LearnerEstimator(OLS(), FoldIndex(), dtype=np.float64)

Est = LearnerEstimator


class Tmp(Est):

    """Temporary class

    Wrapper to get full estimator on no-args instantiation. For compatibility
    with older Scikit-learn versions.
    """

    def __init__(self):
        args = {LearnerEstimator: (OLS(), FullIndex()),
                LayerEnsemble: (FullIndex(), OLS()),
                TransformerEstimator: (Scale(), FullIndex())}[Est]
        super(Tmp, self).__init__(*args)

    __init__.deprecated_original = __init__

    def fit(self, X, y, *args, **kwargs):
        X, y = check_X_y(X, y)
        y = np.asarray(y, X.dtype)
        return super(Tmp, self).fit(X, y, *args, **kwargs)

    def fit_transform(self, X, y, *args, **kwargs):
        X, y = check_X_y(X, y)
        y = np.asarray(y, X.dtype)
        return super(Tmp, self).fit_transform(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        X = check_array(X,)
        return super(Tmp, self).predict(X, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        X = check_array(X)
        return super(Tmp, self).transform(X, *args, **kwargs)


# These are simple run tests to ensure parallel wrapper register backend.
# See parallel for more rigorous tests

def test_learner_fit():
    """[Module | LearnerEstimator] test fit"""
    out = est.fit(X, y)
    assert out is est

    p = est.fit_transform(X, y, refit=False)
    np.testing.assert_array_equal(p, F[:, [0]])


def test_learner_transform():
    """[Module | LearnerEstimator] test transform"""
    p = est.transform(X)
    np.testing.assert_array_equal(p, F[:, [0]])


def test_learner_predict():
    """[Module | LearnerEstimator] test predict"""
    p = est.predict(X)
    np.testing.assert_array_equal(p, P[:, [0]])


def test_learner_clone():
    """[Module | LearnerEstimator] test clone"""
    cl = clone(est)

    p = cl.fit_transform(X, y)
    np.testing.assert_array_equal(p, F[:, [0]])


def test_learner_params_estimator():
    """[Module | LearnerEstimator] test set params on estimator"""
    est.fit(X, y)

    # Just a check that this works
    out = est.get_params()
    assert isinstance(out, dict)

    est.set_params(estimator__offset=10)
    np.testing.assert_warns(ParameterChangeWarning, est.predict, X)


def test_learner_params_indexer():
    """[Module | LearnerEstimator] test set params on indexer"""
    est.fit(X, y)
    est.indexer.set_params(folds=3)
    np.testing.assert_warns(ParameterChangeWarning, est.predict, X)


def test_learner_attr():
    """[Module | LearnerEstimator] test setting attribute"""
    est.fit(X, y)
    est.indexer = FoldIndex(1)
    np.testing.assert_warns(ParameterChangeWarning, est.predict, X)


if run_sklearn:
    def test_learner():
        """[Module | LearnerEstimator] test pass estimator checks"""
        check_estimator(Tmp)
