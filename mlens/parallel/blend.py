"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Estimation engine for parallel preprocessing of blend layer.
"""

from .estimation import BaseEstimator
from .estimation import predict_fold_est, time_
from ..utils import safe_print, print_time
from ..externals.joblib import delayed
from ..externals.sklearn.base import clone


###############################################################################
class Blender(BaseEstimator):

    """Blended fit sub-process class.

    Class for fitting a Layer using Blending.
    """

    def __init__(self, layer, dual=True):
        super(Blender, self).__init__(layer=layer, dual=dual)

    def _format_instance_list(self):
        """Expand the instance lists to every fold with associated indices."""
        e = _expand_instance_list(self.layer.estimators, self.layer.indexer)

        t = _expand_instance_list(self.layer.preprocessing,
                                  self.layer.indexer)

        return e, t

    def _get_col_id(self):
        """Assign unique col_id to every estimator."""
        c = getattr(self.layer, 'classes_', 1)
        return _get_col_idx(self.layer.preprocessing, self.layer.estimators, c)

    def transform(self, X, P, parallel):
        """Predict X.

        Since a blend ensemble does not use folds, transform coincides with
        predict, except that the prediction in fitting is only for a subset
        of X.
        """
        self._check_fitted()

        if self.verbose:
            printout = "stderr" if self.verbose < 50 else "stdout"
            safe_print('Transforming %s' % self.name)
            t0 = time_()

        pred_method = 'predict' if not self.proba else 'predict_proba'

        # Collect estimators - blend only has estimators fitted on 'full'
        # since no folds are used in building the prediction matrix during
        # fitting
        prep, ests = self._retrieve('full')

        parallel(delayed(predict_fold_est)(case=case,
                                           tr_list=prep[case]
                                           if prep is not None else [],
                                           inst_name=est_name,
                                           est=est,
                                           xtest=X,
                                           pred=P,
                                           idx=idx,
                                           name=self.name,
                                           attr=pred_method)
                 for case, (est_name, est, idx) in ests)

        if self.verbose:
            print_time(t0, '%s Done' % self.name, file=printout)


###############################################################################
def _expand_instance_list(instance_list, indexer=None):
    """Build a list of estimation tuples with train and test indices."""
    if instance_list is None or len(instance_list) == 0:
        # Capture cases when there is no preprocessing to avoid running a
        # parallel job.
        return None

    elif isinstance(instance_list, dict):
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
