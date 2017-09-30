"""ML-ENSEMBLE


"""
import numpy as np
from mlens.testing.dummy import OLS, ECM
from mlens.testing import Data

from mlens.ensemble import Subsemble, SuperLearner

data = Data('subsemble', False, False, partitions=2, folds=3)
X, y = data.get_data((30, 4), 3)
data.indexer.fit(X)
(F, wf), (P, wp) = data.ground_truth(X, y, data.indexer.partitions)


def test_subset_fit():
    """[Subsemble] 'fit' and 'predict' runs correctly."""
    meta = OLS()
    meta.fit(F, y)
    g = meta.predict(P)

    ens = Subsemble()
    ens.add(ECM, partitions=2, folds=3, dtype=np.float64)
    ens.add_meta(OLS(), dtype=np.float64)

    ens.fit(X, y)

    pred = ens.predict(X)
    np.testing.assert_array_equal(pred, g)


def test_subset_equiv():
    """[Subsemble] Test equivalence with SuperLearner for J=1."""

    sub = Subsemble(partitions=1)
    sl = SuperLearner()

    sub.add(ECM, dtype=np.float64)
    sl.add(ECM, dtype=np.float64)

    F = sub.fit(X, y).predict(X)
    P = sl.fit(X, y).predict(X)

    np.testing.assert_array_equal(P, F)
