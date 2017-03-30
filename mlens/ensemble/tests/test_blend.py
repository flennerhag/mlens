"""ML-ENSEMBLE

Place holder for more rigorous tests.

"""
import numpy as np
from mlens.metrics import rmse
from mlens.base import BlendIndex
from mlens.utils.dummy import data, ground_truth, OLS, PREPROCESSING, \
    ESTIMATORS, ECM

from mlens.ensemble import BlendEnsemble


FOLDS = 3
LEN = 6
WIDTH = 2
MOD = 2

X, y = data((LEN, WIDTH), MOD)

(F, wf), (P, wp) = ground_truth(X, y, BlendIndex(test_size=3, X=X),
                                'predict', 1, 1, False)


def test_run():
    """[Blend] 'fit' and 'predict' runs correctly."""
    meta = OLS()
    meta.fit(F, y[3:])
    g = meta.predict(P)

    ens = BlendEnsemble(test_size=3)
    ens.add(ESTIMATORS, PREPROCESSING)
    ens.add_meta(OLS())

    ens.fit(X, y)

    pred = ens.predict(X)

    np.testing.assert_array_equal(pred, g)
