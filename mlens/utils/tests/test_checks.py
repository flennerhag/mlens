"""ML-ENSEMBLE

author: Sebastian Flennerhag
:copyirgh: 2017
:licence: MIT
"""
import numpy as np

from mlens.utils.dummy import LAYER, LAYER_CONTAINER
from mlens.utils.formatting import check_instances
from mlens.utils.exceptions import (NotFittedError, LayerSpecificationError,
                                    LayerSpecificationWarning)
from mlens.utils.checks import (check_fit_overlap, check_is_fitted,
                                check_ensemble_build)

from mlens.externals.base import clone


def test_check_is_fitted():
    """[Utils] check_is_fitted : passes fitted."""
    lyr = clone(LAYER)
    lyr.estimators_ = None
    check_is_fitted(lyr, 'estimators_')


def test_check_is_fitted():
    """[Utils] check_is_fitted : raises error."""
    np.testing.assert_raises(NotFittedError,
                             check_is_fitted,
                             LAYER, 'estimators_')


def test_check_ensemble_build_passes():
    """[Utils] check_ensemble_build : passes on default dummy LC."""
    FLAG = check_ensemble_build(LAYER_CONTAINER)
    assert FLAG is True


def test_check_ensemble_build_no_lc():
    """[Utils] check_ensemble_build : raises error on no LC."""
    lc = clone(LAYER_CONTAINER)
    del lc.layers
    np.testing.assert_raises(AttributeError,
                             check_ensemble_build,
                             lc)

def test_check_ensemble_build_lc_None():
    """[Utils] check_ensemble_build : raises error on LC None."""
    lc = clone(LAYER_CONTAINER)
    lc.layers = None
    lc.raise_on_exception = True

    np.testing.assert_raises(LayerSpecificationError,
                             check_ensemble_build,
                             lc)


def test_check_ensemble_build_lc_None_no_raise_():
    """[Utils] check_ensemble_build : raises warning on LC None + no raise_."""
    lc = clone(LAYER_CONTAINER)
    lc.layers = None
    lc.raise_on_exception = False

    FLAG = np.testing.assert_warns(LayerSpecificationWarning,
                                   check_ensemble_build, lc)
    assert FLAG is False


def test_check_estimators():
    """[Base] Test that fitted estimator overlap is correctly checked."""
    fold_fit_incomplete = ['a']
    fold_fit_complete = ['a', 'b']

    full_fit_incomplete = ['a']
    full_fit_complete = ['a', 'b']

    check_fit_overlap(fold_fit_complete, full_fit_complete, 'layer-1')

    try:
        check_fit_overlap(fold_fit_incomplete, full_fit_complete, 'layer-1')
    except ValueError as e:
        assert issubclass(e.__class__, ValueError)
        assert str(e) == \
               ("[layer-1] Not all estimators successfully fitted on "
                "the full data set were fitted during fold predictions. "
                "Aborting.\n[layer-1] Fitted estimators on full data: ['a']\n"
                "[layer-1] Fitted estimators on folds:['a', 'b']")

    try:
        check_fit_overlap(fold_fit_complete, full_fit_incomplete, 'layer-1')
    except ValueError as e:
        assert issubclass(e.__class__, ValueError)
        assert str(e) == \
               ("[layer-1] Not all estimators successfully fitted on the fold "
                "data were successfully fitted on the full data. Aborting.\n"
                "[layer-1] Fitted estimators on full data: ['a', 'b']\n"
                "[layer-1] Fitted estimators on folds:['a']")
