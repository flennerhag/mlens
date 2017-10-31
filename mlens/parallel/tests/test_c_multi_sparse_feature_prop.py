"""
Test specific functionality of the parallel manager module.
"""
import numpy as np
from scipy.sparse import csr_matrix
from mlens.index import FoldIndex
from mlens.ensemble.base import BaseEnsemble
from mlens.externals.sklearn.validation import check_random_state
from mlens.utils.dummy import OLS


SEED = 1324


class TempClass(BaseEnsemble):

    def __init__(self):
        super(TempClass, self).__init__()


def _shuffled(X, y, seed):
    """Shuffle inputs."""
    r = check_random_state(seed)
    idx = r.permutation(y.shape[0])
    return X[idx], y[idx]


class OLSSparse(OLS):

    """Handle Sparse Matrices"""

    def fit(self, X, y):
        return super(OLSSparse, self).fit(X.toarray(), y)

    def predict(self, X):
        return super(OLSSparse, self).predict(X.toarray())


X = np.random.rand(10, 50).astype(np.float32)
y = np.arange(10).astype(np.float32)

first_prop = [1, 2, 3]
n_first_prop = len(first_prop)

second_prop = [i for i in range(n_first_prop)]
second_prop.append(second_prop[-1] + 1)
second_prop.append(second_prop[-1] + 1)
n_second_prop = len(second_prop)

ens1 = TempClass()
ens1.add([OLS(0), OLS(1)], FoldIndex(), propagate_features=first_prop)

ens2 = TempClass()
ens2.add([OLS(0), OLS(1)], FoldIndex(), propagate_features=first_prop)
ens2.add([OLS(2), OLS(3)], FoldIndex(), propagate_features=second_prop)

ens3 = TempClass()
ens3.add([OLSSparse(0), OLSSparse(1)], FoldIndex(), propagate_features=first_prop)
ens3.add([OLSSparse(2), OLSSparse(3)], FoldIndex(), propagate_features=second_prop)

ens4 = TempClass()
ens4.add([OLS(), OLS(1), OLS(2)], FoldIndex(), shuffle=True, random_state=SEED)
ens4.add([OLS(), OLS(1), OLS(2)], FoldIndex(), shuffle=True, random_state=SEED)
ens4.add([OLS(), OLS(1), OLS(2)], FoldIndex(), shuffle=True, random_state=SEED)

ens5 = TempClass()
ens5.add([OLS(), OLS(1), OLS(2)], FoldIndex())


def test_propagation_one():
    """[Parallel] Test feature propagation from original data to first layer"""
    # Check that original data is propagated through first layer
    out_1 = ens1.fit(X, y, return_preds=True)
    np.testing.assert_array_equal(
        out_1[:, :n_first_prop], X[:, first_prop])


def test_propagation_two():
    """[Parallel] Test feature propagation from original data to second layer"""
    # Check that original data is propagated through second layer
    out_2 = ens2.fit(X, y, return_preds=True)
    np.testing.assert_array_equal(
        out_2[:, :n_first_prop], X[:, first_prop])


def test_propagation_three():
    """[Parallel] Test feature propagation from first layer prediction to second layer"""
    out_1 = ens1.fit(X, y, return_preds=True)
    out_2 = ens2.fit(X, y, return_preds=True)
    np.testing.assert_array_equal(
        out_2[:, n_first_prop:n_second_prop], out_1[:, n_first_prop:])


def test_sparse():
    """[Parallel] Test sparse feature propagation."""
    out_1 = ens2.fit(X, y, return_preds=True)
    Z = csr_matrix(X)
    out_2 = ens3.fit(Z, y, return_preds=True)
    np.testing.assert_allclose(
        out_1, out_2.toarray().astype(dtype=np.float32))


def test_shuffle():
    """[Parallel] Test shuffle between layers."""
    h, s = X.copy(), y.copy()
    for i in range(3):
        h, s = _shuffled(h, s, ens4.layers[i].random_state)
        h = ens5.fit(h, s, return_preds=True)

    z = ens4.fit(X, y, return_preds=True)

    np.testing.assert_array_equal(h.astype(np.float32), z)
