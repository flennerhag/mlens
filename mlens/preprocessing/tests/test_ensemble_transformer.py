"""ML-ENSEMBLE

Test ensemble transformer.
"""

import numpy as np
from mlens.utils.dummy import Data, PREPROCESSING
from mlens.utils.dummy import ESTIMATORS, ESTIMATORS_PROBA,ECM, ECM_PROBA

from mlens.preprocessing import EnsembleTransformer


FOLDS = 3
LEN = 12
WIDTH = 2
MOD = 2

ESTS = {(False, False): ECM,
        (False, True): ESTIMATORS,
        (True, False): ECM_PROBA,
        (True, True): ESTIMATORS_PROBA}


def run(cls, proba, preprocessing, **kwargs):
    """Function for executing specified test."""
    if cls == 'subset':
        p = kwargs['n_partitions']
    else:
        p = 1

    ests = ESTS[(proba, preprocessing)]
    prep = PREPROCESSING if preprocessing else None

    data = Data(cls, proba, preprocessing, **kwargs)

    X, y = data.get_data((LEN, WIDTH), MOD)
    (F, wf), _ = data.ground_truth(X, y, p)

    ens = EnsembleTransformer()
    ens.add(cls, ests, prep, proba=proba, dtype=np.float64, **kwargs)
    ens.fit(X, y)

    pred = ens.transform(X)

    np.testing.assert_array_equal(F, pred)


def test_stack_run():
    """[EnsembleTransformer | Stack | Prep] retrieves fit predictions."""
    run('stack', False, True, n_splits=3)


def test_stack_run_no_prep():
    """[EnsembleTransformer | Stack | No Prep] retrieves fit predictions."""
    run('stack', False, False, n_splits=3)


def test_stack_run_proba():
    """[EnsembleTransformer | Stack | Prep] retrieves fit predictions."""
    run('stack', True, True, n_splits=3)


def test_stack_run_no_prep_proba():
    """[EnsembleTransformer | Stack | No Prep] retrieves fit predictions."""
    run('stack', True, False, n_splits=3)


def test_blend_run():
    """[EnsembleTransformer | Blend | Prep] retrieves fit predictions."""
    run('blend', False, True, test_size=0.4)


def test_blend_run_no_prep():
    """[EnsembleTransformer | Blend | No Prep] retrieves fit predictions."""
    run('blend', False, False, test_size=0.4)


def test_blend_run_proba():
    """[EnsembleTransformer | Blend | Prep] retrieves fit predictions."""
    run('blend', True, True, test_size=0.4)


def test_blend_run_no_prep_proba():
    """[EnsembleTransformer | Blend | No Prep] retrieves fit predictions."""
    run('blend', True, False, test_size=0.4)


def test_subset_run():
    """[EnsembleTransformer | Subset | Prep] retrieves fit predictions."""
    run('subset', False, True, n_partitions=2, n_splits=2)


def test_subset_run_no_prep():
    """[EnsembleTransformer | Subset | No Prep] retrieves fit predictions."""
    run('subset', False, False, n_partitions=2, n_splits=2)


def test_subset_run_proba():
    """[EnsembleTransformer | Subset | Prep] retrieves fit predictions."""
    run('subset', True, True, n_partitions=2, n_splits=2)


def test_subset_run_no_prep_proba():
    """[EnsembleTransformer | Subset | No Prep] retrieves fit predictions."""
    run('subset', True, False, n_partitions=2, n_splits=2)
