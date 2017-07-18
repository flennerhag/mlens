"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Estimation engine for parallel preprocessing of blend layer.
"""
from ._base_functions import predict_fold_est, fit, predict, construct_args
from .estimation import BaseEstimator
from ..externals.joblib import delayed
from ..externals.sklearn.base import clone

from copy import deepcopy


FUNCS = {'fit': fit,
         'predict': predict,
         'predict_proba': predict,
         }


###############################################################################
class Blender(BaseEstimator):

    """Blended fit sub-process class.

    Class for fitting a Layer using Blending.
    """

    def __init__(self, job, layer, n):
        super(Blender, self).__init__(layer=layer)
        self.dir = job.dir

        self.execute = FUNCS[job.j] if job.j != 'transform' else transform
        self.args = construct_args(self.execute, job, n)

    def run(self, parallel):
        """Execute stacking."""
        super(Blender, self).run(parallel)

    def _format_instance_list(self):
        """Expand the instance lists to every fold with associated indices."""
        self.e = _expand_instance_list(self.layer.estimators,
                                       self.layer.indexer)

        self.t = _expand_instance_list(self.layer.preprocessing,
                                       self.layer.indexer)

    def _get_col_id(self):
        """Assign unique col_id to every estimator."""
        c = getattr(self.layer, 'classes_', 1)
        k = self.layer.n_feature_prop
        self.c = _get_col_idx(self.layer.preprocessing, self.layer.estimators,
                              c, k)

    def _build_scores(self, s):
        """Build a cv-score mapping."""
        scores = dict()
        for k, v in s:
            case_name, est_name = k.split('___')

            if case_name == '':
                name = est_name
            else:
                name = '%s__%s' % (case_name, est_name)

            scores[name] = (v, 0.)  # mean, std
        return scores


def transform(inst, X, P, parallel):
    """Predict X.

    Since a blend ensemble does not use folds, transform coincides with
    predict, except that the prediction in fitting is only for a subset
    of X.
    """
    inst._check_fitted()
    pred_method = inst.layer._predict_attr

    # Collect estimators - blend only has estimators fitted on 'full'
    # since no folds are used in building the prediction matrix during fitting
    prep, ests = inst._retrieve('full')

    parallel(delayed(predict_fold_est)(tr_list=deepcopy(prep[case])
                                       if prep is not None else [],
                                       est=est,
                                       x=X,
                                       pred=P,
                                       idx=idx,
                                       attr=pred_method)
             for case, (_, est, idx) in ests)


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


def _get_col_idx(preprocessing, estimators, labels, n_feature_prop):
    """Utility for assigning each ``est`` in each ``prep`` a unique ``col_id``.

    Parameters
    ----------
    preprocessing : dict
        dictionary of preprocessing cases.

    estimators : dict
        dictionary of lists of estimators per preprocessing case.

    labels : int
        number of labels to expand col_id with

    n_feature_prop : int
        number of features being propagated. Predictions are concatenated from
        the right.
    """
    inc = 1 if labels is None else labels

    if isinstance(preprocessing, list) or preprocessing is None:
        # Simple iteration of list
        idx = {(None, inst_name): int(n_feature_prop + inc * i)
               for i, (inst_name, _) in enumerate(estimators)}
    else:
        # Nested for loop required
        case_list, idx, col = sorted(preprocessing), dict(), n_feature_prop

        for case in case_list:
            for inst_name, _ in estimators[case]:
                idx[case, inst_name] = col
                col += inc
    return idx
