"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Estimation engine for parallel preprocessing of stacked layer.
"""

from .estimation import BaseEstimator
from ..externals.sklearn.base import clone


###############################################################################
class Stacker(BaseEstimator):

    """Stacked fit sub-process class.

    Class for fitting a Layer using Stacking.
    """

    def __init__(self, job, layer, n):
        super(Stacker, self).__init__(layer=layer)
        self._default_initialization(job, n)

    def run(self, parallel):
        """Execute stacking."""
        super(Stacker, self).run(parallel)

    def _format_instance_list(self):
        """Expand the instance lists to every fold with associated indices."""
        self.e = _expand_instance_list(self.layer.estimators,
                                       self.layer.indexer)

        self.t = _expand_instance_list(self.layer.preprocessing,
                                       self.layer.indexer)

    def _get_col_id(self):
        """Assign unique col_id to every estimator."""
        c = getattr(self.layer, 'classes_', 1)
        p = len(self.layer.cases)
        k = self.layer.n_feature_prop
        self.c = _get_col_idx(self.e, p, c, k)


###############################################################################
def _expand_instance_list(instance_list, indexer):
    """Build a list of fold-specific estimation tuples w. train and test idx.

    The full learner library is copied for each fold and used for building
    the Z matrix of dimensions n * (L), where n is the number of training
    samples and L number of base learners.

    Examples
    --------
    Passing a list estimators

    >>> import numpy as np
    >>> from mlens.utils.dummy import OLS
    >>> from mlens.base import FoldIndex
    >>> from mlens.parallel.stack import _expand_instance_list
    >>> X = np.arange(12)
    >>> indexer = FoldIndex(3, X=X)
    >>> instance_list = [('%i' % i, OLS()) for i in range(2)]
    >>> _expand_instance_list(instance_list, indexer)
    [(None, None, None, [('0', OLS(offset=0)), ('1', OLS(offset=0))]),
     (None,
      ((4, 12),),
      (0, 4),
      [('0__f0', OLS(offset=0)), ('1__f0', OLS(offset=0))]),
     (None,
      ((0, 4), (8, 12)),
      (4, 8),
      [('0__f1', OLS(offset=0)), ('1__f1', OLS(offset=0))]),
     (None,
      ((0, 8),),
      (8, 12),
      [('0__f2', OLS(offset=0)), ('1__f2', OLS(offset=0))])]

    Passing a dict estimators per cases

    >>> import numpy as np
    >>> from mlens.utils.dummy import OLS
    >>> from mlens.base import FoldIndex
    >>> from mlens.parallel.stack import _expand_instance_list
    >>> X = np.arange(12)
    >>> indexer = FoldIndex(3, X=X)
    >>> instance_list = {'a': [('%i' % i, OLS()) for i in range(2)],
    ...                  'b': [('%i' % i, OLS(1)) for i in range(1)]}
    >>> _expand_instance_list(instance_list, indexer)
    [list of estimation tuples, beginning with main estimators]
    """
    splits = indexer.n_splits

    if instance_list is None or len(instance_list) == 0:
        # Capture cases when there is no preprocessing to avoid running a
        # parallel job.
        return None

    elif isinstance(instance_list, dict):
        # We need to build fit list on a case basis

        # --- Full data ---
        # Estimators to be fitted on full data. List entries have format:
        # (case, no_train_idx, no_test_idx, est_list)
        # Each est_list have entries (inst_name, cloned_est)
        ls = [('%s' % case, None, None,
               [(n, clone(e)) for n, e in instance_list[case]])
              for case in sorted(instance_list)]

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (case__fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        if indexer is not None:
            fd = [('%s__f%i' % (case, i % splits),
                   tri,
                   tei,
                   [('%s__f%i' % (n, i % splits), clone(e)) for n, e in
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
        ls = [(None, None, None, [(n, clone(e)) for n, e in instance_list])]

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (None, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        if indexer is not None:
            ls.extend([(None,
                        tri,
                        tei,
                        [('%s__f%i' % (n, i % splits), clone(e)) for n, e in
                         instance_list])
                       for i, (tri, tei) in enumerate(indexer.generate())
                       ])
    return ls


def _get_col_idx(instance_list, n_main, labels, n_feature_prop):
    """Utility for assigning columns ids to each fold-specific estimator.

    Parameters
    ----------
    instance_list : list
        list of instances per case and per cv fold

    n_main : int
        number of main cases.

    labels : int
        number of labels to expand col_id with

    n_feature_prop : int
        number of features being propagated. Predictions are concatenated from
        the right.
    """
    inc = 1 if labels is None else labels

    # Set up estimator column mapping
    # We select the main estimators by filtering out
    # fold-specific estimators and assigning each of the main ests a col_id
    idx, col = dict(), n_feature_prop
    for meta_name, _, _, estimator_list in instance_list[:n_main]:
        for est_name, _ in estimator_list:
            idx[(meta_name, est_name)] = col

            col += inc

    # Map every fold-specific estimator back onto the just created column
    # mapping for the final estimators. The fold-specific estimators should
    # have the same col_id as the main estimators.
    for meta_name_w_fold, _, _, estimator_list in instance_list[n_main:]:

        # 'meta_name__f0' > 'meta_name'
        try:
            # Fails if meta_name is None
            meta_name = '__'.join(meta_name_w_fold.split('__')[:-1])
        except AttributeError:
            meta_name = None

        # Assign a column to estimators in belonging to the case__fold entry
        for est_name_w_fold, _ in estimator_list:

            # 'est_name__f0' > 'est_name'
            est_name = '__'.join(est_name_w_fold.split('__')[:-1])

            idx[meta_name_w_fold, est_name_w_fold] = idx[meta_name, est_name]

    return idx
