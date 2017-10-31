"""ML-ENSEMBLE

:author: Sebastian Flennerhag
"""

import numpy as np
from mlens.preprocessing import Subset, Shift
from mlens.testing.dummy import Data

X, _ = Data('stack', False, False).get_data((10, 4), 2)

sub = Subset([0, 1])

def test_subset_1():
    """[Preprocessing | Subset]: assert correct subset."""
    assert sub.fit_transform(X).shape[1] == 2


def test_subset_2():
    """[Preprocessing | Subset]: assert X is returned for empty subset."""
    sub.set_params(**{'subset': None})
    out = sub.fit_transform(X)
    assert id(out) == id(X)


def test_shift():
    """[Preprocessing | Shift] test lagging."""
    sh = Shift(2)

    sh.fit(X)

    Z = sh.transform(X)

    assert Z.shape[0] + 2 ==  X.shape[0]
    np.testing.assert_array_equal(Z, X[:-2])
