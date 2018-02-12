"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Indexer checks
"""
from __future__ import division

from numbers import Integral
import warnings


def check_full_index(n_samples, folds, raise_on_exception):
    """Check that folds can be constructed from passed arguments."""
    if not isinstance(folds, Integral):
        raise ValueError("'folds' must be an integer. "
                         "type(%s) was passed." % type(folds))

    if folds <= 1:
        if raise_on_exception:
            raise ValueError("Need at least 2 folds to partition data. "
                             "Got %i." % folds)
        else:
            if folds == 1:
                warnings.warn("'folds' is 1, will return full index as "
                              "both training set and test set.")
    if folds > n_samples:
        raise ValueError("Number of splits %i is greater than the number "
                         "of samples: %i." % (folds, n_samples))


def check_partial_index(n_samples, test_size, train_size, n_test, n_train):
    """Check that folds can be constructed from passed arguments."""
    if n_test + n_train > n_samples:
        raise ValueError("The selection of train (%r) and test (%r) samples "
                         "lead to a subsets greater than the number of "
                         "observations (%i). Implied test size: %i, "
                         "implied train size: "
                         "%i." % (test_size, train_size,
                                  n_samples, n_test, n_train))

    for n, i, j in zip(('test', 'train'),
                       (n_test, n_train),
                       (test_size, train_size)):
        if n == 0:
            raise ValueError("The %s set size is 0 with current selection ("
                             "%r): "
                             "cannot create %s subset. Assign a greater "
                             "proportion of samples to the %s set (total "
                             "samples size: %i)." % (i, j, i, i, n_samples))

    if n_samples < 2:
        raise ValueError("Sample size < 2: nothing to create subset from.")


def check_subsample_index(n_samples, partitions, folds, raise_):
    """Check input validity of the SubsampleIndexer."""
    if not isinstance(partitions, Integral):
        raise ValueError("'partitions' must be an integer. "
                         "type(%s) was passed." % type(partitions))

    if not partitions > 0:
        raise ValueError("'partitions' must be a positive integer. "
                         "{} was passed.".format(partitions))

    if not isinstance(folds, Integral):
        raise ValueError("'folds' must be an integer. "
                         "type(%s) was passed." % type(folds))

    if folds == 1:
        if raise_ or partitions > 1:
            raise ValueError("Need at least 2 folds for splitting partitions. "
                             "Got %i." % folds)
        else:
            if partitions == 1 and folds == 1:
                warnings.warn("'folds' is 1, will return full index as "
                              "both training set and test set.")
    s = partitions * folds
    if s > n_samples:
        raise ValueError("Number of total splits %i is greater than the "
                         "number of samples: %i." % (s, n_samples))
