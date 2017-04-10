"""ML-ENSEMBLE


"""
import numpy as np
from mlens.utils.dummy import OLS, ECM
from mlens.base import SubsetIndex

from mlens.parallel.subset import _expand_instance_list, _get_col_idx

from mlens.ensemble import Subsemble, SuperLearner

X = np.arange(24).reshape((12, 2))
y = X[:, 0] * X[:, 1]

estimators = [('ols-%i' % i, OLS(i)) for i in range(2)]
indexer = SubsetIndex(2, 3, X=X)


def ground_truth():
    """Ground truth for subset test.
    """
    e = _expand_instance_list(estimators, indexer)

    P = np.zeros((12, 2 * 2))
    F = np.zeros((12, 2 * 2))

    cols = _get_col_idx(e, 2, 1)

    for name, tri, tei, est_list in e:
        for est_name, est in est_list:
            if tei is None:
                est.fit(X[tri[0]:tri[1]], y[tri[0]:tri[1]])
                p = est.predict(X)
                P[:, cols[(name, est_name)]] = p
                continue

            ti = np.hstack([np.arange(t0, t1) for t0, t1 in tri])
            te = np.hstack([np.arange(t0, t1) for t0, t1 in tei])
            col = cols[(name, est_name)]

            est.fit(X[ti], y[ti])
            p = est.predict(X[te])
            F[te, col] = p
    return F, P

F, P = ground_truth()


def test_subset_fit():
    """[Subsemble] 'fit' and 'predict' runs correctly."""
    meta = OLS()
    meta.fit(F, y)
    g = meta.predict(P)

    ens = Subsemble()
    ens.add(estimators, partitions=2, folds=3)
    ens.add_meta(OLS())

    ens.fit(X, y)

    pred = ens.predict(X)

    np.testing.assert_array_equal(pred, g)


def test_subset_equiv():
    """[Subsemble] Test equivalence with SuperLearner for J=1."""

    sub = Subsemble(partitions=1)
    sl = SuperLearner()

    sub.add(ECM)
    sl.add(ECM)

    F = sub.fit(X, y).predict(X)
    P = sl.fit(X, y).predict(X)

    np.testing.assert_array_equal(P, F)
