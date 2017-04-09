"""ML-ENSEMBLE


"""

import numpy as np
from mlens import metrics


y = np.ones(10)
p = np.arange(10)


def test_rmse():
    """[Metrics] rmse."""
    z  = metrics.rmse(y, p)
    np.testing.assert_equal(np.array(z), np.array(4.5276925690687087))


def test_mape():
    """[Metrics] mape."""
    z = metrics.mape(y, p)
    np.testing.assert_equal(np.array(z), np.array(3.7))


def test_wape():
    """[Metrics] mape."""
    z = metrics.wape(y, p)
    np.testing.assert_equal(np.array(z), np.array(3.7))
