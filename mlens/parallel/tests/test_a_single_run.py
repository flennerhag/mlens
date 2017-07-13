"""ML-ENSEMBLE

"""

import numpy as np

from mlens.utils.dummy import ECM, Data
from mlens.ensemble.base import LayerContainer
from mlens.externals.sklearn.base import clone


X, y = Data('stack', False, False).get_data((6, 2), 2)

lc = LayerContainer()
lc.add(ECM, 'full', dtype=np.float64)

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
    return

    F = get_gt()

    lc.fit(X, y)

    out = lc.predict(X)

    np.testing.assert_array_equal(F, out)
