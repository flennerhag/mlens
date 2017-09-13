"""ML-ENSEMBLE

"""

import numpy as np

from mlens.utils.dummy import ECM, Data
from mlens.ensemble.base import LayerContainer
from mlens.externals.sklearn.base import clone


X, y = Data('stack', False, False).get_data((6, 2), 2)

lc1 = LayerContainer()
lc1.add(ECM, 'full', meta=True, dtype=np.float64)

lc2 = LayerContainer()
lc2.add(ECM, 'full', meta=False, dtype=np.float64)


def get_gt():
    """Build ground truth."""

    F = np.empty((X.shape[0], len(ECM)))

    for i, (_, est) in enumerate(ECM):
        e = clone(est)

        assert e is not est

        e.fit(X, y)

        F[:, i] = e.predict(X)

    return F


def test_single_run():
    """[Parallel | Single Run] Test single run routine."""
    F = get_gt()

    lc1.fit(X, y)
    out = lc1.predict(X)

    np.testing.assert_array_equal(F, out)

def test_ringle_run_no_meta():
    """[Parallel | Single Run] Check single run predictions in fit call."""
    F = get_gt()

    out = lc2.fit(X, y, return_preds=True)[-1]

    np.testing.assert_array_equal(F, out)
