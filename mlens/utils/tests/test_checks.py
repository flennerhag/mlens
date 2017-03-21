"""ML-ENSEMBLE

author: Sebastian Flennerhag
:copyirgh: 2017
:licence: MIT
"""

from mlens.utils.formatting import check_instances
from mlens.utils.checks import check_fit_overlap




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
