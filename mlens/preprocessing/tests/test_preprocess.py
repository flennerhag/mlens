"""ML-ENSEMBLE

:author: Sebastian Flennerhag
"""

from __future__ import division, print_function

from mlens.preprocessing import Subset
import numpy as np

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


sub = Subset([0, 1])

def test_subset_1():
    """[Preprocessing] Subset: assert correct subset."""
    assert sub.fit_transform(X).shape[1] == 2


def test_subset_2():
    """[Preprocessing] Subset: assert X is returned for empty subset."""
    sub.set_params(**{'subset': None})
    out = sub.fit_transform(X)
    assert id(out) == id(X)
