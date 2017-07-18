"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Estimation engine for parallel preprocessing of subsemble layer.
"""

from .estimation import BaseEstimator
from ..externals.sklearn.base import clone


###############################################################################
class SubStacker(BaseEstimator):

    """Stacked subset fit sub-process class.

    Class for fitting a Layer using Subsemble.
    """

    def __init__(self, job, layer, n):
        super(SubStacker, self).__init__(layer=layer)
        self._default_initialization(job, n)

    def run(self, parallel):
        """Execute subsembling"""
        super(SubStacker, self).run(parallel)

    def _format_instance_list(self):
        """Expand the instance lists to every fold with associated indices."""
        self.e = _expand_instance_list(self.layer.estimators,
                                       self.layer.indexer)

        self.t = _expand_instance_list(self.layer.preprocessing,
                                       self.layer.indexer)

    def _get_col_id(self):
        """Assign unique col_id to every estimator."""
        c = getattr(self.layer, 'classes_', 1)
        p = len(self.layer.cases) * self.layer.indexer.n_partitions
        k = self.layer.n_feature_prop
        self.c = _get_col_idx(self.e, p, c, k)


###############################################################################
def _expand_instance_list(instance_list, indexer):
    """Build a list of subset-specific estimation tuples w. train and test idx.

    The subset's ``_expand_insance_list`` function expands a list of
    base learners in two dimensions:
        1. Partitions
        2. Folds

    For each partition, the full learner library is copied as final estimators
    on that partition. The full learner library is then copied again for
    each fold within that partition, and these estimators are used for building
    the Z matrix of dimensions n * (J*L), where n is the number of training
    samples, J the number of partitions, and L number of base learners.

    Examples
    --------
    Passing a list estimators

    >>> import numpy as np
    >>> from mlens.utils.dummy import OLS
    >>> from mlens.base import SubsetIndex
    >>> from mlens.parallel.subset import _expand_instance_list
    >>> X = np.arange(12)
    >>> indexer = SubsetIndex(3, X=X)
    >>> instance_list = [('%i', OLS()) for i in range(2)]
    >>> _expand_instance_list(instance_list, indexer)
    [list of estimation tuples, beginning with main estimators]

    Passing a dict estimators per cases

    >>> import numpy as np
    >>> from mlens.utils.dummy import OLS
    >>> from mlens.base import SubsetIndex
    >>> from mlens.parallel.subset import _expand_instance_list
    >>> X = np.arange(12)
    >>> indexer = SubsetIndex(3, X=X)
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
        ls = [('%s__j%i' % (case, j), partition, None,
               [(n, clone(e)) for n, e in instance_list[case]])
              for case in sorted(instance_list)
              for j, partition in enumerate(indexer.partition())]

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (case__fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        if indexer is not None:
            fd = [('%s__j%i__f%i' % (case, i // splits, i % splits),
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
        ls = [('j%i' % i, partition, None,
               [(n, clone(e)) for n, e in instance_list])
              for i, partition in enumerate(indexer.partition())]

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        if indexer is not None:
            ls.extend([('j%i__f%i' % (i // splits, i % splits),
                        tri,
                        tei,
                        [('%s__f%i' % (n, i % splits), clone(e)) for n, e in
                         instance_list])
                       for i, (tri, tei) in enumerate(indexer.generate())
                       ])
    return ls


def _get_col_idx(instance_list, n_main, labels, n_feature_prop):
    """Utility for assigning columns ids to each subset-specific estimator.

    Parameters
    ----------
    instance_list : list
        list of instances per case and per cv fold

    n_main : int
        number of main cases. Either ``n_partitions`` or
        ``n_partitions * n_cases``.

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
    # mapping for the final estimators:
    # the fold-specific estimators in a partition j and fold f should have
    # the same col_id as the main estimators for partition j.
    for meta_name_w_fold, _, _, estimator_list in instance_list[n_main:]:

        # 'case__j0__f0' > 'case__j0' or 'j0__f0' > 'j0
        meta_name = '__'.join(meta_name_w_fold.split('__')[:-1])

        # Assign a column to estimators in belonging to the case__fold entry
        for est_name_w_fold, _ in estimator_list:

            # 'est_name__j0__f0' > 'est_name__j0'
            est_name = '__'.join(est_name_w_fold.split('__')[:-1])

            idx[meta_name_w_fold, est_name_w_fold] = idx[meta_name, est_name]

    return idx
