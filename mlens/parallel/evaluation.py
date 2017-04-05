"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Cross-validation jobs for an :class:`Evaluator` instance.
"""

from .estimation import fit_trans, _slice_array, _transform_tr, _fit_est, \
    _predict_est
from ..utils import pickle_load
from ..externals.sklearn.base import clone

import os
from joblib import delayed

try:
    from time import perf_counter as time
except ImportError:
    from time import time


def _preprocess():
    """Fit preprocessors."""
    pass


class Evaluation(object):

    """Evaluation engine.

    Run a job for an :class:`Evaluator` instance.

    Parameters
    ----------
    evaluator : :class:`Evaluator`
        Evaluator instance to run job for.
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def preprocess(self, parallel, X, y, dir):
        """Fit preprocessing pipelines.

        Fit all preprocessing pipelines in parallel and store as a
        ``preprocessing_`` attribute on the :class:`Evaluator`.

        Parameters
        ----------
        parallel : :class:`joblib.Parallel`
            The instance to use for parallel fitting.

        X : array-like of shape [n_samples, n_features]
            Training set to use for estimation. Can be memmaped.

        y : array-like of shape [n_samples, ]
            labels for estimation. Can be memmaped.

        dir : directory of cache to dump fitted transformers before assembly.
        """
        preprocessing = _expand_instance_list(self.evaluator.preprocessing,
                                              self.evaluator.indexer)

        parallel(delayed(fit_trans)(dir=dir,
                                    case=case,
                                    inst=instance_list,
                                    X=X,
                                    y=y,
                                    idx=tri,
                                    name=None)
                 for case, tri, _, instance_list in preprocessing)

        self.evaluator.preprocessing_ = \
            [(tup[0], pickle_load(os.path.join(dir, '%s__t' % tup[0])))
             for tup in preprocessing]

    def evaluate(self, parallel, X, y, dir):
        """cross-validation of estimators.

        Parameters
        ----------
        parallel : :class:`joblib.Parallel`
            The instance to use for parallel fitting.

        X : array-like of shape [n_samples, n_features]
            Training set to use for estimation. Can be memmaped.

        y : array-like of shape [n_samples, ]
            labels for estimation. Can be memmaped.

        dir : directory of cache to dump fitted transformers before assembly.
        """
        preprocessing = dict(getattr(self.evaluator, 'preprocessing_', []))
        estimators = _expand_instance_list(self.evaluator.estimators,
                                           self.evaluator.indexer)

        scores = parallel(delayed(fit_score)(
                case=case,
                tr_list=preprocessing[case],
                est_name=est_name,
                est=est,
                params=(i, params),
                X=X,
                y=y,
                idx=(tri, tei),
                scorer=self.evaluator.scorer,
                error_score=self.evaluator.error_score)
                          for case, tri, tei, est_list in estimators
                          for est_name, est in est_list
                          for i, params in
                          enumerate(self.evaluator.params[
                                        (case.split('__')[0]
                                         if case is not None else None,
                                         est_name.split('__')[0])]))
        self.evaluator.scores_ = scores


###############################################################################
def fit_score(case, tr_list, est_name, est, params, X, y, idx, scorer,
              error_score):
    """Fit and score an estimator given a set of params."""
    raise_ = error_score is None

    # Fit estimator
    est = clone(est).set_params(**params[1])

    xtrain, ytrain, _ = _slice_array(X, y, idx[0])

    for tr_name, tr in tr_list:
        xtrain = _transform_tr(xtrain, tr, tr_name, case, None)

    t0 = time()
    est = _fit_est(xtrain, ytrain, est, raise_, est_name, case, None)
    fit_time = time() - t0

    # Predict and score
    xtest, ytest, _ = _slice_array(X, y, idx[1])

    for tr_name, tr in tr_list:
        xtest = _transform_tr(xtest, tr, tr_name, case, None)

    scores = []
    for y, x in zip([ytrain, ytest], [xtrain, xtest]):
        p = _predict_est(x, est, raise_, est_name, case, None, 'predict')

        try:
            s = scorer(y, p)
        except Exception:
            s = error_score

        scores.append(s)

    return case, est_name, params[0], scores[0], scores[1], fit_time


###############################################################################
def _expand_instance_list(instance_list, indexer):
    """Build a list of fold-specific estimation tuples w. train and test idx.

    The full learner library is copied for each fold and used for building
    the Z matrix of dimensions n * (L), where n is the number of training
    samples and L number of base learners.

    See Also
    --------
    :obj:`mlens.parallel.stack._expand_instance_list`
    """
    splits = indexer.n_splits
    if isinstance(instance_list, dict):
        # We need to build fit list on a case basis

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (case__fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        return [('%s__f%i' % (case, i % splits),
                 tri,
                 tei,
                 [('%s__f%i' % (n, i % splits), clone(e)) for n, e in
                  instance_list[case]])
                for case in sorted(instance_list)
                for i, (tri, tei) in enumerate(indexer.generate())
                ]
    else:
        # No cases to worry about: expand the list of named instance tuples

        # Estimators to be fitted on each fold. List entries have format:
        # (None, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        return [(None,
                 tri,
                 tei,
                 [('%s__f%i' % (n, i % splits), clone(e)) for n, e in
                  instance_list])
                for i, (tri, tei) in enumerate(indexer.generate())
                ]
