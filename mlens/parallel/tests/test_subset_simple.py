import numpy as np
from mlens.utils.dummy import OLS
from mlens.base import SubsetIndex

from mlens.parallel.subset import _expand_instance_list, _get_col_idx

from mlens.ensemble.base import LayerContainer

x = np.arange(24).reshape((12, 2))
y = x[:, 0] * x[:, 1]

estimators = [('ols-%i' % i, OLS(i)) for i in range(2)]
indexer = SubsetIndex(2, 3, X=x)


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
                est.fit(x[tri[0]:tri[1]], y[tri[0]:tri[1]])
                p = est.predict(x)
                P[:, cols[(name, est_name)]] = p
                continue

            ti = np.hstack([np.arange(t0, t1) for t0, t1 in tri])
            te = np.hstack([np.arange(t0, t1) for t0, t1 in tei])
            col = cols[(name, est_name)]

            est.fit(x[ti], y[ti])
            p = est.predict(x[te])
            F[te, col] = p
    return F, P

F, P = ground_truth()


def test_subset_fit_predict_transform():
    """[Parallel | Subset | No Prep]: second test of fit, predict transform."""
    lc = LayerContainer().add(estimators=estimators,
                              cls='subset',
                              proba=False,
                              indexer=indexer,
                              preprocessing=None)

    f = lc.fit(x, y, return_preds=-1)[-1]
    p = lc.predict(x)
    t = lc.transform(x)

    np.testing.assert_array_equal(f, F)
    np.testing.assert_array_equal(p, P)
    np.testing.assert_array_equal(t, F)
