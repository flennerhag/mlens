"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Classes for partitioning training data.
"""

import numpy as np
import warnings


class FullIndex(object):

    """Indexer that generates the full size of X.

    K-Fold iterator that generates fold index tuples.

    FullIndex creates a generator that returns a tuple of stop and start
    positions to be used for numpy array slicing [stop:start]. Note that
    slicing works well for the test set, but for the training set it is
    recommended to concatenate the index for training data that comes before
    the current test set with the index for the training data that comes after.
    This can easily be achieved with::

        for train_tup, test_tup in self.generate():
            train_slice = numpy.hstack([numpy.arange(t0, t1) for t0, t1 in
                                      train_tup])

            xtrain, xtest = X[train_slice], X[test_tup[0]:test_tup[1]]

    Warnings
    --------
    Slicing only works for the test set, which is convex, but typically not for
    the training set. To build get a training index, use
    ``hstack([np.arange(t0, t1) for t0, t1 in tri])``.

    Examples
    --------

    Creating arrays of folds and checking overlap

    >>> import numpy as np
    >>> from mlens.base.indexer import FullIndex
    >>> X = np.arange(10)
    >>> print("Data set: %r" % X)
    >>> print()
    >>>
    >>> idx = FullIndex(X, 4)
    >>>
    >>> for train, test in idx.generate(as_array=True):
    ...     print('TRAIN IDX: %32r | TEST IDX: %16r' % (train, test))
    >>>
    >>> print()
    >>>
    >>> for train, test in idx.generate(as_array=True):
            print('TRAIN SET: %32r | TEST SET: %16r' % (X[train], X[test]))
    >>>
    >>> for train_idx, test_idx in idx.generate(as_array=True):
    ...     assert not any([i in X[test_idx] for i in X[train_idx]])
    >>>
    >>>print()
    >>>
    >>> print("No overlap between train set and test set.")
    Data set: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    TRAIN IDX:     array([3, 4, 5, 6, 7, 8, 9]) | TEST IDX: array([0, 1, 2])
    TRAIN IDX:     array([0, 1, 2, 6, 7, 8, 9]) | TEST IDX: array([3, 4, 5])
    TRAIN IDX:  array([0, 1, 2, 3, 4, 5, 8, 9]) | TEST IDX:    array([6, 7])
    TRAIN IDX:  array([0, 1, 2, 3, 4, 5, 6, 7]) | TEST IDX:    array([8, 9])

    TRAIN SET:     array([3, 4, 5, 6, 7, 8, 9]) | TEST SET: array([0, 1, 2])
    TRAIN SET:     array([0, 1, 2, 6, 7, 8, 9]) | TEST SET: array([3, 4, 5])
    TRAIN SET:  array([0, 1, 2, 3, 4, 5, 8, 9]) | TEST SET:    array([6, 7])
    TRAIN SET:  array([0, 1, 2, 3, 4, 5, 6, 7]) | TEST SET:    array([8, 9])

    No overlap between train set and test set.

    Passing only one fold with raise_on_exception set to False

    >>> import numpy as np
    >>> from mlens.base.indexer import FullIndex
    >>> X = np.arange(3)
    >>> print("Data set: %r" % X)
    >>> print()
    >>>
    >>> idx = FullIndex(X, 1, raise_on_exception=False)
    >>>
    >>> for train, test in idx.generate(as_array=True):
    ...     print('TRAIN IDX: %10r | TEST IDX: %10r' % (train, test))
    Data set: array([0, 1, 2])

    /../mlens/base/indexer.py:167: UserWarning: 'n_splits' is 1, will return
    full index as both training set and test set.
    warnings.warn("'n_splits' is 1, will return full index as "

    Data set: array([0, 1, 2])
    TRAIN IDX: array([0, 1, 2]) | TEST IDX: array([0, 1, 2])
    """

    def __init__(self, n_splits=2, X=None, raise_on_exception=True):
        self.n_splits = n_splits
        self.raise_on_exception = raise_on_exception

        if X is not None:
            self.fit(X)

    def fit(self, X):
        """Set indexer up for slicing an array of length X."""
        n = X.shape[0]
        check_index(n, self.n_splits, self.raise_on_exception)

        self.n_samples = n

    def _gen_indices(self):
        n_samples = self.n_samples
        n_splits = self.n_splits

        if n_splits == 1:
            # Return the full index as both training and test set
            yield ((0, n_samples),),  (0, n_samples)
        else:

            # Get the length of the test set. If n_samples mod n_splits is
            #  not 0, increment n_remainder folds by 1
            tei_len = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)
            tei_len[:n_samples % n_splits] += 1

            last = 0  # counter for where last test index stopped
            for size in tei_len:
                tei_start, tei_stop = last, last + size
                tri_start_below, tri_stop_below = 0, tei_start
                tri_start_above, tri_stop_above = tei_stop, n_samples

                # Build test set tuple
                tei = (tei_start, tei_stop)

                # Build train set tuple(s)
                # Check if the first tuple is (0, 0) - if so drop
                if tri_start_below == tri_stop_below:
                    tri = ((tri_start_above, tri_stop_above),)

                # Check if the first tuple is (n, n) - if so drop
                elif tri_start_below == tri_stop_below:
                    tri = ((tri_start_above, tri_stop_above),)

                else:
                    tri = ((tri_start_below, tri_stop_below),
                           (tri_start_above, tri_stop_above))

                yield (tri, tei)
                last = tei_stop

    def generate(self, X=None, as_array=False):
        """Generator."""
        # Check that the instance have some array information to work with
        if not hasattr(self, 'n_samples'):
            if X is None:
                raise AttributeError("No array provided to indexer. Either "
                                     "pass an array to the 'generate' method, "
                                     "or call the 'fit' method first or "
                                     "initiate the instance with an array X "
                                     "as argument.")
            else:
                # Need to call fit to continue
                self.fit(X)

        for tri, tei in self._gen_indices():

            # return np.arrays
            if as_array:
                tri = np.hstack([np.arange(t0, t1) for t0, t1 in tri])
                tei = np.arange(tei[0], tei[1])

            yield tri, tei


def check_index(n_samples, n_splits, raise_on_exception):
    """Check that folds can be constructed from passed arguments."""
    if not isinstance(n_splits, int):
        raise ValueError("'n_splits' must be an integer. "
                         "type(%s) was passed." % type(n_splits))

    if n_splits <= 1:
        if raise_on_exception:
            raise ValueError("Need at least 2 folds to partition data. "
                             "Got %i." % n_splits)
        else:
            if n_splits == 1:
                warnings.warn("'n_splits' is 1, will return full index as "
                              "both training set and test set.")
    if n_splits > n_samples:
        raise ValueError("Number of splits %i is greater than the number "
                         "of samples: %i." % (n_splits, n_samples))
