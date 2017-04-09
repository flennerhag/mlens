"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Estimation engine for parallel preprocessing of estimators in a single run,
such as when fitting a final layer (meta estimator) that does not require
propagating predictions.
"""

from .estimation import BaseEstimator
from ..externals.sklearn.base import clone


###############################################################################
class SingleRun(BaseEstimator):

    """Single run fit sub-process class.

    Class for fitting a estimators in a layer without any sub-fits.
    """

    def __init__(self, layer, dual=True):
        super(SingleRun, self).__init__(layer=layer, dual=dual)

    def _format_instance_list(self):
        """Expand the instance lists to every fold with associated indices."""
        e = _expand_instance_list(self.layer.estimators)
        t = _expand_instance_list(self.layer.preprocessing)

        return e, t

    def _get_col_id(self):
        """Assign unique col_id to every estimator."""
        c = getattr(self.layer, 'classes_', 1)
        return _get_col_idx(self.layer.preprocessing, self.layer.estimators, c)


###############################################################################
def _expand_instance_list(instance_list):
    """Build a list of estimation tuples with train and test indices."""
    # We modify the instance list slightly by adding None for the
    # training and test set indices
    if isinstance(instance_list, dict):
        return [(case, None, None,
                 [(n, clone(e)) for n, e in instance_list[case]])
                for case in sorted(instance_list)]
    else:
        return [(None, None, None,
                 [(n, clone(e)) for n, e in instance_list])]


def _get_col_idx(preprocessing, estimators, labels):
    """Utility for assigning each ``est`` in each ``prep`` a unique ``col_id``.

    Parameters
    ----------
    preprocessing : dict or list
        mapping of preprocessing cases, if any.

    estimators : dict or list
        mapping of estimators per preprocessing case, or list of estimators.
    """
    inc = 1 if labels is None else labels

    # Set up main columns mapping
    if isinstance(preprocessing, list) or preprocessing is None:
        idx = {(None, inst_name): int(inc * i) for i, (inst_name, _) in
               enumerate(estimators)}
    else:
        # Nested for loop required
        case_list, idx, col = sorted(preprocessing), dict(), 0

        for case in case_list:
            for inst_name, _ in estimators[case]:
                idx[case, inst_name] = col
                col += inc

    return idx
