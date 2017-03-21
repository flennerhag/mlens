"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Classes for partitioning training data.
"""

from abc import abstractmethod
from numbers import Integral
import numpy as np
import warnings


class BaseIndex(object):
    """Base Index class."""

    @abstractmethod
    def fit(self, X):
        """Method for storing array data.

        Parameters
        ----------
        X : array-like
            array to collect dimension data from.

        Returns
        -------
        instance
        """

    @abstractmethod
    def _gen_indices(self):
        """Method for constructing the index generator.

        Returns
        -------
        iterable
        """

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

            if as_array:
                # return np.arrays
                if isinstance(tri[0], tuple):
                    # If a tuple of indices, build iteratively
                    tri = np.hstack([np.arange(t0, t1) for t0, t1 in tri])
                else:
                    tri = np.arange(tri[0], tri[1])

                tei = np.arange(tei[0], tei[1])

            yield tri, tei


class BlendIndex(BaseIndex):

    """Indexer that generates two non-overlapping subsets of X.

    Iterator that generates one training fold and one test fold that are
    non-overlapping and that may or may not partition all of X depending on the
    user's specification.

    BlendIndex creates a singleton generator (has on iteration) that
    yields two tuples of ``(start, stop)`` integers that can be used for
    numpy array slicing (i.e. ``X[stop:start]``). If a full array index
    is desired this can easily be achieved with::

        for train_tup, test_tup in self.generate():
            train_slice = numpy.hstack([numpy.arange(t0, t1) for t0, t1 in
                                      train_tup])

            test_slice = numpy.hstack([numpy.arange(t0, t1) for t0, t1 in
                                      test_tup])

    See Also
    --------
    :class:`FullIndex`

    Examples
    --------

    Selecting an absolute test size, with train size as the remainder

    >>> import numpy as np
    >>> from mlens.base.indexer import BlendIndex
    >>> X = np.arange(8)
    >>> idx = BlendIndex(3)
    >>> print('Test size: 3')
    >>> for tri, tei in idx.generate(X):
    ...     print('TEST (idx | array): (%i, %i) | %r ' % (tei[0], tei[1],
    ...                                                   X[tei[0]:tei[1]]))
    ...     print('TRAIN (idx | array): (%i, %i) | %r ' % (tri[0], tri[1],
    ...                                                    X[tri[0]:tri[1]]))
    Test size: 3
    TEST (idx | array): (5, 8) | array([5, 6, 7])
    TRAIN (idx | array): (0, 5) | array([0, 1, 2, 3, 4])

    Selecting a test and train size less than the total

    >>> import numpy as np
    >>> from mlens.base.indexer import BlendIndex
    >>> X = np.arange(8)
    >>> idx = BlendIndex(3, 4, X)
    >>> print('Test size: 3')
    >>> print('Train size: 4')
    >>> for tri, tei in idx.generate(X):
    ...     print('TEST (idx | array): (%i, %i) | %r ' % (tei[0], tei[1],
    ...                                                   X[tei[0]:tei[1]]))
    ...     print('TRAIN (idx | array): (%i, %i) | %r ' % (tri[0], tri[1],
    ...                                                    X[tri[0]:tri[1]]))
    Test size: 3
    Train size: 4
    TEST (idx | array): (4, 7) | array([4, 5, 6])
    TRAIN (idx | array): (0, 4) | array([0, 1, 2, 3])

    Selecting a percentage of observations as test and train set

    >>> import numpy as np
    >>> from mlens.base.indexer import BlendIndex
    >>> X = np.arange(8)
    >>> idx = BlendIndex(0.25, 0.45, X)
    >>> print('Test size: 25% * 8 = 2')
    >>> print('Train size: 45% * 8 < 4 -> 3')
    >>> for tri, tei in idx.generate(X):
    ...     print('TEST (idx | array): (%i, %i) | %r ' % (tei[0], tei[1],
    ...                                                   X[tei[0]:tei[1]]))
    ...     print('TRAIN (idx | array): (%i, %i) | %r ' % (tri[0], tri[1],
    ...                                                    X[tri[0]:tri[1]]))
    Test size: 25% * 8 = 2
    Train size: 50% * 8 < 4 ->
    TEST (idx | array): (3, 5) | array([[3, 4]])
    TRAIN (idx | array): (0, 3) | array([[0, 1, 2]])
    """

    def __init__(self, test_size, train_size=None,
                 X=None, raise_on_exception=True):

        self.test_size = test_size
        self.train_size = train_size
        self.raise_on_exception = raise_on_exception

        if X is not None:
            self.fit(X)

    def fit(self, X):
        """Set indexer up for slicing an array of length X."""
        self.n_samples = X.shape[0]

        # Get number of test samples
        if isinstance(self.test_size, Integral):
            self.n_test = self.test_size
        else:
            self.n_test = int(np.floor(self.test_size * self.n_samples))

        # Get number of train samples
        if self.train_size is None:
            # Partition X - we coerce a positive value here:
            # if n_test is oversampled will get at final check
            self.n_train = int(np.abs(self.n_samples - self.n_test))

        elif isinstance(self.train_size, Integral):
            self.n_train = self.train_size

        else:
            self.n_train = int(np.floor(self.train_size * self.n_samples))

        _check_partial_index(self.n_samples, self.test_size, self.train_size,
                             self.n_test, self.n_train)

        return self

    def _gen_indices(self):
        yield (0, self.n_train), (self.n_train, self.n_train + self.n_test)


class FullIndex(BaseIndex):

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

    See Also
    --------
    :class:`BlendIndex`

    Examples
    --------

    Creating arrays of folds and checking overlap

    >>> import numpy as np
    >>> from mlens.base.indexer import FullIndex
    >>> X = np.arange(10)
    >>> print("Data set: %r" % X)
    >>> print()
    >>>
    >>> idx = FullIndex(4, X)
    >>>
    >>> for train, test in idx.generate(as_array=True):
    ...     print('TRAIN IDX: %32r | TEST IDX: %16r' % (train, test))
    >>>
    >>> print()
    >>>
    >>> for train, test in idx.generate(as_array=True):
    ...     print('TRAIN SET: %32r | TEST SET: %16r' % (X[train], X[test]))
    >>>
    >>> for train_idx, test_idx in idx.generate(as_array=True):
    ...     assert not any([i in X[test_idx] for i in X[train_idx]])
    >>>
    >>> print()
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
    >>> idx = FullIndex(1, X, raise_on_exception=False)
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
        _check_full_index(n, self.n_splits, self.raise_on_exception)

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


###############################################################################
def _check_full_index(n_samples, n_splits, raise_on_exception):
    """Check that folds can be constructed from passed arguments."""
    if not isinstance(n_splits, Integral):
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


def _check_partial_index(n_samples, test_size, train_size,
                         n_test, n_train):
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
