"""ML-ENSEMBLE

:author: Sebastian Flennerhag
"""

from __future__ import division, print_function

from mlens.preprocessing import Subset, StandardScaler
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler as StandardScaler_

# training data
np.random.seed(100)
X = np.random.random((10, 5))

# noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

# Change scales
X[:, 0] *= 10
X[:, 1] += 10
X[:, 2] *= 5
X[:, 3] *= 3
X[:, 4] /= 10

sc1 = StandardScaler()
sc2 = StandardScaler_()

sub1 = Subset([1, 2])
sub2 = Subset()


def test_standard_scaler_df():
    """[Preprocessing] StandardScaler: test against sklearn parent class."""
    Z = DataFrame(X)
    Zout = sc1.fit_transform(Z)
    Xout = sc2.fit_transform(X)
    Xout = DataFrame(Xout)
    np.array_equal(Xout, Zout)


def test_subset_1():
    """[Preprocessing] Subset: assert correct subset."""
    assert sub1.fit_transform(X).shape[1] == 2


def test_subset_2():
    """[Preprocessing] Subset: assert X is returned for empty subset."""
    out = sub2.fit_transform(X)
    assert id(out) == id(X)
