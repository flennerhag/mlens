"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Estimation engine for parallel preprocessing of stacked layer.
"""

from ..externals.sklearn.base import clone
from .estimation import BaseEstimator


###############################################################################
class Stacker(BaseEstimator):
    """Stacked fit sub-process class.

    Class for fitting a Layer using Stacking.
    """

    def __init__(self, layer, dual=True):
        super(Stacker, self).__init__(layer=layer, dual=dual)

    def _format_instance_list(self):
        """Expand the instance lists to every fold with associated indices."""
        e = _expand_instance_list(self.layer.estimators, self.layer.indexer)

        t = _expand_instance_list(self.layer.preprocessing,
                                  self.layer.indexer)

        return e, t

    def _get_col_id(self):
        """Assign unique col_id to every estimator."""
        c = getattr(self.layer, 'classes_', 1)
        return _get_col_idx(self.layer.preprocessing,
                            self.layer.estimators,
                            self.e, c)


###############################################################################
def _expand_instance_list(instance_list, indexer=None):
    """Build a list of estimation tuples with train and test indices."""
    ls = list()

    if isinstance(instance_list, dict):
        # We need to build fit list on a case basis

        # --- Full data ---
        # Estimators to be fitted on full data. List entries have format:
        # (case, no_train_idx, no_test_idx, est_list)
        # Each est_list have entries (inst_name, cloned_est)
        ls.extend([(case, None, None,
                    [(n, clone(e)) for n, e in instance_list[case]])
                   for case in sorted(instance_list)])

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (case__fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        if indexer is not None:
            fd = [('%s__%i' % (case, i),
                   tri,
                   tei,
                   [('%s__%i' % (n, i), clone(e)) for n, e in
                    instance_list[case]])
                  for case in sorted(instance_list)
                  for i, (tri, tei) in enumerate(indexer.generate())
                  ]
            ls.extend(fd)

    else:
        # No cases to worry about: expand the list of named instance tuples

        # --- Full data ---
        # Estimators to be fitted on full data. List entries have format:
        # (no_case, no_train_idx, no_test_idx, est_list)
        # Each est_list have entries (inst_name, cloned_est)
        ls.extend([(None, None, None,
                    [(n, clone(e)) for n, e in instance_list])])

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        if indexer is not None:
            ls.extend([('%i' % i,
                        tri,
                        tei,
                        [('%s__%i' % (n, i), clone(e)) for n, e in
                         instance_list])
                       for i, (tri, tei) in enumerate(indexer.generate())
                       ])

    return ls


def _get_col_idx(preprocessing, estimators, estimator_folds, labels):
    """Utility for assigning each ``est`` in each ``prep`` a unique ``col_id``.

    Parameters
    ----------
    preprocessing : dict
        dictionary of preprocessing cases.

    estimators : dict
        dictionary of lists of estimators per preprocessing case.

    estimator_folds : list
        list of estimators per case and per cv fold

    """
    inc = 1 if labels is None else labels

    # Set up main columns mapping
    if isinstance(preprocessing, list) or preprocessing is None:
        idx = {(None, inst_name): int(inc * i) for i, (inst_name, _) in
               enumerate(estimators)}

        case_list = [None]
    else:
        # Nested for loop required
        case_list, idx, col = sorted(preprocessing), dict(), 0

        for case in case_list:
            for inst_name, _ in estimators[case]:
                idx[case, inst_name] = col
                col += inc

    # Map every estimator-case-fold entry back onto the just created column
    # mapping for estimators
    for tup in estimator_folds:
        if tup[0] in case_list:
            # A main estimator, will not be used for folded predictions
            continue

        # Get case name from the name_id entry
        # With cases, names are in the form (case_name__fold_num)
        # Otherwise named as (fold_num) - in this case the case name is None
        case = tup[0].split('__')[0] if '__' in tup[0] else None

        # Assign a column to estimators in belonging to the case__fold entry
        for inst_name_fold_num, _ in tup[-1]:
            inst_name = inst_name_fold_num.split('__')[0]
            idx[tup[0], inst_name_fold_num] = idx[case, inst_name]

    return idx
