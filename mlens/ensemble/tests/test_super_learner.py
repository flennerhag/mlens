"""ML-ENSEMBLE

Place holder for more rigorous tests.

"""
import numpy as np
from mlens.metrics import rmse
from mlens.base import FoldIndex
from mlens.utils.dummy import data, ground_truth, OLS, PREPROCESSING, \
    ESTIMATORS, ECM

from mlens.ensemble import SuperLearner


FOLDS = 3
LEN = 6
WIDTH = 2
MOD = 2

X, y = data((LEN, WIDTH), MOD)

(F, wf), (P, wp) = ground_truth(X, y, FoldIndex(n_splits=FOLDS, X=X),
                                'predict', 1, 1, False)


def test_run():
    """[SuperLearner] 'fit' and 'predict' runs correctly."""
    meta = OLS()
    meta.fit(F, y)
    g = meta.predict(P)

    ens = SuperLearner(folds=FOLDS)
    ens.add(ESTIMATORS, PREPROCESSING)
    ens.add_meta(OLS())

    ens.fit(X, y)

    pred = ens.predict(X)

    np.testing.assert_array_equal(pred, g)
