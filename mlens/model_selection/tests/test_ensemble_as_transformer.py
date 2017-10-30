"""ML-ENSEMBLE

Test ensemble transformer.
"""
import os
import numpy as np
from mlens.testing.dummy import Data, PREPROCESSING
from mlens.testing.dummy import ESTIMATORS, ESTIMATORS_PROBA,ECM, ECM_PROBA
from mlens.model_selection import EnsembleTransformer
from mlens.ensemble import SequentialEnsemble

try:
    from contextlib import redirect_stdout, redirect_stderr
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stdout
    redirect_stderr = redirect_stdout


FOLDS = 3
LEN = 12
WIDTH = 2
MOD = 2

ESTS = {(False, False): ECM,
        (False, True): ESTIMATORS,
        (True, False): ECM_PROBA,
        (True, True): ESTIMATORS_PROBA}


def run(cls, kls, proba, preprocessing, **kwargs):
    """Function for executing specified test."""
    model_selection = kwargs.pop('model_selection', None)
    if kls == 'subsemble':
        p = kwargs['partitions']
    else:
        p = 1

    ests = ESTS[(proba, preprocessing)]
    prep = PREPROCESSING if preprocessing else None

    data = Data(kls, proba, preprocessing, **kwargs)

    X, y = data.get_data((LEN, WIDTH), MOD)
    (F, wf), _ = data.ground_truth(X, y, p)

    with open(os.devnull, 'w') as f, redirect_stderr(f):
        ens = cls()
        ens.add(kls, ests, prep, proba=proba, dtype=np.float64, **kwargs)

        if model_selection:
            ens.model_selection = True

        ens.fit(X, y)

        pred, _ = ens.transform(X, y)

    np.testing.assert_array_equal(F, pred)


def test_et_stack_run():
    """[EnsembleTransformer | Stack | Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'stack', False, True, folds=3)


def test_et_stack_run_no_prep():
    """[EnsembleTransformer | Stack | No Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'stack', False, False, folds=3)


def test_et_stack_run_proba():
    """[EnsembleTransformer | Stack | Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'stack', True, True, folds=3)


def test_et_stack_run_no_prep_proba():
    """[EnsembleTransformer | Stack | No Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'stack', True, False, folds=3)


def test_et_blend_run():
    """[EnsembleTransformer | Blend | Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'blend', False, True, test_size=0.4)


def test_et_blend_run_no_prep():
    """[EnsembleTransformer | Blend | No Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'blend', False, False, test_size=0.4)


def test_et_blend_run_proba():
    """[EnsembleTransformer | Blend | Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'blend', True, True, test_size=0.4)


def test_et_blend_run_no_prep_proba():
    """[EnsembleTransformer | Blend | No Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'blend', True, False, test_size=0.4)


def test_et_subset_run():
    """[EnsembleTransformer | Subset | Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'subsemble', False, True, partitions=2, folds=2)


def test_et_subset_run_no_prep():
    """[EnsembleTransformer | Subset | No Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'subsemble', False, False, partitions=2, folds=2)


def test_et_subset_run_proba():
    """[EnsembleTransformer | Subset | Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'subsemble', True, True, partitions=2, folds=2)


def test_et_subset_run_no_prep_proba():
    """[EnsembleTransformer | Subset | No Prep] retrieves fit predictions."""
    run(EnsembleTransformer, 'subsemble', True, False, partitions=2, folds=2)



def test_stack_run():
    """[SequentialEnsemble | Stack | Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'stack', False, True, folds=3, model_selection=True)


def test_stack_run_no_prep():
    """[SequentialEnsemble | Stack | No Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'stack', False, False, folds=3, model_selection=True)


def test_stack_run_proba():
    """[SequentialEnsemble | Stack | Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'stack', True, True, folds=3, model_selection=True)


def test_stack_run_no_prep_proba():
    """[SequentialEnsemble | Stack | No Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'stack', True, False, folds=3, model_selection=True)


def test_blend_run():
    """[SequentialEnsemble | Blend | Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'blend', False, True, test_size=0.4, model_selection=True)


def test_blend_run_no_prep():
    """[SequentialEnsemble | Blend | No Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'blend', False, False, test_size=0.4, model_selection=True)


def test_blend_run_proba():
    """[SequentialEnsemble | Blend | Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'blend', True, True, test_size=0.4, model_selection=True)


def test_blend_run_no_prep_proba():
    """[SequentialEnsemble | Blend | No Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'blend', True, False, test_size=0.4, model_selection=True)


def test_subset_run():
    """[SequentialEnsemble | Subset | Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'subsemble', False, True, partitions=2, folds=2, model_selection=True)


def test_subset_run_no_prep():
    """[SequentialEnsemble | Subset | No Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'subsemble', False, False, partitions=2, folds=2, model_selection=True)


def test_subset_run_proba():
    """[SequentialEnsemble | Subset | Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'subsemble', True, True, partitions=2, folds=2, model_selection=True)


def test_subset_run_no_prep_proba():
    """[SequentialEnsemble | Subset | No Prep] retrieves fit predictions."""
    run(SequentialEnsemble, 'subsemble', True, False, partitions=2, folds=2, model_selection=True)
