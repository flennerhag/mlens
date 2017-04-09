"""ML-ENSEMBLE

"""
import os
import numpy as np
from mlens.parallel.estimation import _load_trans
from mlens.utils.exceptions import ParallelProcessingError
import warnings

def test_load_transformer():
    """[Parallel | Estimation] test load transformer block."""

    f = os.path.join(os.getcwd(), 'dummy')

    with warnings.catch_warnings(record=True) as w:
        np.testing.assert_raises(ParallelProcessingError,
                                 _load_trans, f, 'test', (0.1, 0.2), False)

    assert len(w) == 1
