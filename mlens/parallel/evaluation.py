"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Cross-validation jobs for an :class:`Evaluator` instance.
"""

import os
import warnings

from ._base_functions import fit_trans, _slice_array, _transform
from ..externals.joblib import delayed
from ..utils import pickle_load
from ..utils.exceptions import FitFailedWarning
from ..externals.sklearn.base import clone


try:
    from time import perf_counter as time
except ImportError:
    from time import time


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
                                    x=X,
                                    y=y,
                                    idx=tri)
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

        dir : str
            directory of cache to dump fitted transformers before assembly.
        """
        preprocessing = dict(getattr(self.evaluator, 'preprocessing_', []))
        estimators = _expand_instance_list(self.evaluator.estimators,
                                           self.evaluator.indexer)

        scores = parallel(delayed(fit_score)(
                case=case,
                tr_list=preprocessing[case] if case in preprocessing else [],
                est_name=est_name,
                est=est,
                params=(i, params),
                x=X,
                y=y,
                idx=(tri, tei),
                scorer=self.evaluator.scorer,
                error_score=self.evaluator.error_score)
                          for case, tri, tei, est_list in estimators
                          for est_name, est in est_list
                          for i, params in
                          enumerate(
                                  self.evaluator.params[_name(case, est_name)]
                                    ))
        self.evaluator.scores_ = scores


###############################################################################
def _name(case, est_name):
    """Get correct param_dict name."""
    if case is not None:
        return case.split('__')[0], est_name.split('__')[0]
    else:
        return est_name.split('__')[0]


def fit_score(case, tr_list, est_name, est, params, x, y, idx, scorer,
              error_score):
    """Wrapper around fit function to determine how to handle exceptions."""

    if error_score is None:
        # If fit or scoring fails, we raise errors.
        return _fit_score(case, tr_list, est_name, est, params, x, y, idx,
                          scorer, error_score)

    else:
        # Otherwise, we issue a warning and set an error score.
        try:
            return _fit_score(case, tr_list, est_name, est, params, x, y, idx,
                              scorer, error_score)
        except Exception as exception:

            warnings.warn("Cross validation failed. Setting error score {}"
                          ".".format(error_score), FitFailedWarning)

            return case, est_name, params[0], error_score, error_score, 0


def _fit_score(case, tr_list, est_name, est, params, x, y, idx, scorer,
               error_score):
    """Fit an estimator and generate scores for train and test set."""
    est = clone(est).set_params(**params[1])

    # Prepare training set
    xtrain, ytrain = _slice_array(x, y, idx[0])

    for tr_name, tr in tr_list:
        xtrain, ytrain = _transform(tr, xtrain, ytrain)

    # We might have to rebase the training labels since a BlendEnsemble would
    # make xtrain. Since Blend is sequential, we can discard the first 'n'
    # observation until the dimensions match
    rebase = ytrain.shape[0] - xtrain.shape[0]
    ytrain = ytrain[rebase:]

    # Fit estimator
    t0 = time()
    est = est.fit(xtrain, ytrain)
    fit_time = time() - t0

    # Prepare test set
    xtest, ytest = _slice_array(x, y, idx[1])

    for tr_name, tr in tr_list:
        xtest = tr.transform(xtest)

    # Score train and test set
    train_score = scorer(est, xtrain, ytrain)
    test_score = scorer(est, xtest, ytest)

    return case, est_name, params[0], train_score, test_score, fit_time


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
