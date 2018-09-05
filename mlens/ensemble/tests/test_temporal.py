"""ML-ENSEMBLE

Place holder for more rigorous tests.

"""
import numpy as np
from mlens.metrics import rmse
from mlens.testing.dummy import Data, ESTIMATORS, PREPROCESSING, OLS

from mlens.ensemble import TemporalEnsemble

LEN = 20
WIDTH = 2
MOD = 2

data = Data('temporal', False, True, step_size=5, window=10, lag=2)
X, y = data.get_data((LEN, WIDTH), MOD)

(F, wf), (P, wp) = data.ground_truth(X, y)


def test_run():
    """[Blend] 'fit' and 'predict' runs correctly."""
    meta = OLS()
    meta.fit(F, y[5:])
    g = meta.predict(P)

    ens = TemporalEnsemble(step_size=5, window=10, lag=2)
    ens.add(ESTIMATORS, PREPROCESSING, dtype=np.float64)
    ens.add(OLS(), meta=True, dtype=np.float64)

    ens.fit(X, y)

    pred = ens.predict(X)
    np.testing.assert_array_equal(pred, g)
