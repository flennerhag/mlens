"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Estimation engine for parallel preprocessing of blend layer.
"""

from .base import BaseEstimator
from ..externals.base import clone


###############################################################################
class Blender(BaseEstimator):
    """Blended fit sub-process class.

    Class for fitting a Layer using Blending.
    """

    def __init__(self, layer, labels=None, dual=True):
        super(Blender, self).__init__(layer=layer, labels=labels, dual=dual)

    def _format_instance_list(self):
        """Expand the instance lists to every fold with associated indices."""
        e = _expand_instance_list(self.layer.estimators, self.layer.indexer)

        t = _expand_instance_list(self.layer.preprocessing,
                                  self.layer.indexer)

        return e, t

    def _get_col_id(self, labels):
        """Assign unique col_id to every estimator."""
        return _get_col_idx(self.layer.preprocessing, self.layer.estimators,
                            labels)


###############################################################################
def _expand_instance_list(instance_list, indexer=None):
    """Build a list of estimation tuples with train and test indices."""
    if isinstance(instance_list, dict):
        # List entries have format:
        # (case, train_idx, test_idx, est_list)
        # Each est_list have entries (est_name, cloned_est)
        if indexer is not None:
            return [('%s' % case, tri, tei,
                     [('%s' % n, clone(e)) for n, e in instance_list[case]])
                    for case in sorted(instance_list)
                    for tri, tei in indexer.generate()
                    ]
    else:
        # No cases to worry about: expand the list of named instance tuples

        # List entries have format:
        # ('inst', train_idx, test_idx, est_list)
        # Each est_list have entries (est_name, cloned_est)
        if indexer is not None:
            return [(None, tri, tei,
                     [('%s' % n, clone(e)) for n, e in instance_list])
                    for tri, tei in indexer.generate()
                    ]


def _get_col_idx(preprocessing, estimators, labels):
    """Utility for assigning each ``est`` in each ``prep`` a unique ``col_id``.

    Parameters
    ----------
    preprocessing : dict
        dictionary of preprocessing cases.

    estimators : dict
        dictionary of lists of estimators per preprocessing case.
    """
    inc = 1 if labels is None else labels

    if isinstance(preprocessing, list) or preprocessing is None:
        # Simple iteration of list
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
