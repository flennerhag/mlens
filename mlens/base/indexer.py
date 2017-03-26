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


def _prune_train(start_below, stop_below, start_above, stop_above):
    """Checks if indices above or below are empty (i, i) and removes them."""
    if start_below == stop_below:
        tri = ((start_above, stop_above),)

    elif start_above == stop_above:
        tri = ((start_below, stop_below),)

    else:
        tri = ((start_below, stop_below), (start_above, stop_above))
    return tri


def _partition(n, p):
    """Get partition sizes for a given number of samples and partitions.

    This method will split n samples into p partitions evenly sized partitions.
    If there is a remainder from the split, the r first folds will be
    incremented by 1.

    Parameters
    ----------
    n : int
        number of samples

    p : int
        number of partitions
    """
    sizes = (n // p) * np.ones(p, dtype=np.int)
    sizes[:n % p] += 1
    return sizes


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

        Default is the standard K-Fold.

        Returns
        -------
        iterable
        """
        n_samples = getattr(self, 'n_samples')
        n_splits = getattr(self, 'n_splits')

        if n_splits == 1:
            # Return the full index as both training and test set
            yield ((0, n_samples),), (0, n_samples)
        else:
            # Get the length of the test sets
            tei_len = _partition(n_samples, n_splits)

            last = 0
            for size in tei_len:

                # Test set
                tei_start, tei_stop = last, last + size
                tei = (tei_start, tei_stop)

                # Train set
                tri_start_below, tri_stop_below = 0, tei_start
                tri_start_above, tri_stop_above = tei_stop, n_samples

                tri = _prune_train(tri_start_below, tri_stop_below,
                                   tri_start_above, tri_stop_above)

                yield tri, tei
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

            if as_array:
                tri = self._build_range(tri)
                tei = self._build_range(tei)

            yield tri, tei

    @staticmethod
    def _build_range(idx):
        if isinstance(idx[0], tuple):
            return np.hstack([np.arange(t0, t1) for t0, t1 in idx])
        else:
            return np.arange(idx[0], idx[1])


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
    :class:`FoldIndex`

    Examples
    --------

    Selecting an absolute test size, with train size as the remainder

    >>> import numpy as np
    >>> from mlens.base.indexer import BlendIndex
    >>> X = np.arange(8)
    >>> idx = BlendIndex(3, rebase=False)
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

    Rebasing the test set to be 0-indexed

    >>> import numpy as np
    >>> from mlens.base.indexer import BlendIndex
    >>> X = np.arange(8)
    >>> idx = BlendIndex(3, rebase=True)
    >>> print('Test size: 3')
    >>> for tri, tei in idx.generate(X):
    ...     print('TEST tuple: (%i, %i) | array: %r' % (tei[0], tei[1],
    ...                                                 np.arange(tei[0],
    ...                                                           tei[1])))
    Test size: 3
    TEST tuple: (0, 3) | array: array([0, 1, 2])
    """

    def __init__(self, test_size=0.5, train_size=None, rebase=True,
                 X=None, raise_on_exception=True):

        self.test_size = test_size
        self.train_size = train_size
        self.rebase = rebase
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

        self.n_test_samples = self.n_test

        return self

    def _gen_indices(self):
        if self.rebase:
            yield (0, self.n_train), (0, self.n_test)
        else:
            yield (0, self.n_train), (self.n_train, self.n_train + self.n_test)


class FoldIndex(BaseIndex):

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
    >>> from mlens.base.indexer import FoldIndex
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
    >>> from mlens.base.indexer import FoldIndex
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

        self.n_test_samples = self.n_samples = n

        return self

    def _gen_indices(self):
        """Generate K-Fold iterator."""
        return super(FoldIndex, self)._gen_indices()


class SubSampleIndexer(BaseIndex):

    r"""Subsample index generator.

    Generates a cross-validation folds according to the following strategy:
        1. split ``X`` into ``J`` partitions
        2. for each partition
            (a) for each fold v, create train index of all idx not in v
            (b) concatenate all the fold v indices into a test index for \
                fold v that spans all partitions.

    If ``J`` is set to 1, the SubSampleIndexer reduces to the
    :class:`FullIndexer`, which returns standard K-Fold train and test set
    indices.

    See Also
    --------
    :class:`FoldIndex`

    References
    ----------
    .. [1] Sapp, S., van der Laan, M. J., & Canny, J. (2014).
    Subsemble: an  ensemble method for combining subset-specific algorithm
    fits. Journal of Applied Statistics, 41(6), 1247–1259.
    http://doi.org/10.1080/02664763.2013.864263

    Parameters
    ----------
    n_partitions : int (default = 2)
        Number of partitions to split data in. If ``n_partitions=1``,
        :class:`SubsambleIndex` reduces to standard K-Fold.

    n_splits : int (default = 2)
        Number of splits to create in each partition. ``n_splits`` can
        not be
        1 if ``n_partition > 1``. Note that if ``n_splits = 1``, both the
        train and test set will index the full data.

    X : array-like of shape [n_samples,] , optional
        the training set to partition. The training label array is also,
        accepted, as only the first dimension is used. If ``X`` is not
        passed
        at instantiation, the ``fit`` method must be called before
        ``generate``, or ``X`` must be passed as an argument of
        ``generate``.

    raise_on_exception : bool (default = True)
        whether to warn on suspicious slices or raise an error.

    Examples
    --------
    >>> import numpy as np
    >>> from mlens.base import SubSampleIndexer
    >>> X = np.arange(10)
    >>> idx = SubSampleIndexer(3, X=X)
    >>>
    >>> print('Partitions of X:')
    >>> print('J = 1: %r' % X[0:4])
    >>> print('J = 2: %r' % X[4:7])
    >>> print('J = 3: %r' % X[7:9])
    >>> print()
    >>> print('SubsampleIndexer splits:')
    >>> for i, (tri, tei) in enumerate(idx.generate()):
    ...     fold = i % 2 + 1
    ...     part = i // 2 + 1
    ...     train = np.hstack([np.arange(t0, t1) for t0, t1 in tri])
    ...     test = np.hstack([np.arange(t0, t1) for t0, t1 in tei])
    >>>     print("J = %i | v = %i | "
    ...           "train: %15r | test: % r" % (part, fold, train, test))
    Partitions of X:
    J = 1: array([0, 1, 2, 3])
    J = 2: array([4, 5, 6])
    J = 3: array([7, 8])

    SubsampleIndexer splits:
    J = 1 | v = 1 | train:   array([2, 3]) | test: array([0, 1, 4, 5, 7, 8])
    J = 1 | v = 2 | train:   array([0, 1]) | test: array([2, 3, 6, 9])
    J = 2 | v = 1 | train:      array([6]) | test: array([0, 1, 4, 5, 7, 8])
    J = 2 | v = 2 | train:   array([4, 5]) | test: array([2, 3, 6, 9])
    J = 3 | v = 1 | train:      array([9]) | test: array([0, 1, 4, 5, 7, 8])
    J = 3 | v = 2 | train:   array([7, 8]) | test: array([2, 3, 6, 9])
    """

    def __init__(self, n_partitions=2, n_splits=2,
                 X=None, raise_on_exception=True):

        self.n_partitions = n_partitions
        self.n_splits = n_splits
        self.raise_on_exception = raise_on_exception
        if X is not None:
            self.fit(X)

    def fit(self, X):
        """Set indexer up for slicing an array of length X."""
        n = X.shape[0]
        _check_subsample_index(n, self.n_partitions, self.n_splits,
                               self.raise_on_exception)

        self.n_samples = self.n_test_samples = n
        return self

    def partition(self, X=None):
        """Get partition indices for training full subset estimators."""
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

        if self.n_partitions == 1:
            # Return a None index
            return
        else:
            # Return the partition indices.
            p_len = _partition(self.n_samples, self.n_partitions)
            p_last = 0
            for p_size in p_len:
                yield p_last, p_last + p_size
                p_last += p_size

    def _build_test_sets(self):
        """Build global test folds for each split of every partition."""
        n_partitions = self.n_partitions
        n_samples = self.n_samples
        n_splits = self.n_splits

        if n_partitions == 1:
            # This corresponds to the FullIndexer case
            return
        else:

            # --- Create global test set folds ---
            # In each partition, the test set spans all partitions
            # Hence we must run through all partitions once first to
            # register
            # the global test set fold for each split of the n_splits

            # Partition sizes
            p_len = _partition(n_samples, n_partitions)

            # Since the splitting is sequential and deterministic,
            # we build a
            # list of global test set indices. Hence, for any split, the
            # test index will be a tuple of indexes of test folds from each
            # partition. By concatenating these slices, the full test fold
            # is constructed.
            tei = [[] for _ in range(n_splits)]
            p_last = 0
            for p_size in p_len:
                p_start, p_stop = p_last, p_last + p_size

                t_len = _partition(p_stop - p_start, n_splits)

                # Append the partition's test fold indices to the
                # global directory for that fold number.
                t_last = p_start
                for i, t_size in enumerate(t_len):
                    t_start, t_stop = t_last, t_last + t_size

                    tei[i] += [(t_start, t_stop)]
                    t_last += t_size

                p_last += p_size

        return tei

    def _gen_indices(self):
        n_partitions = self.n_partitions
        n_samples = self.n_samples
        n_splits = self.n_splits

        T = self._build_test_sets()

        if T is None:
            # Standard FoldIndex case
            super(SubSampleIndexer, self)._gen_indices()
        else:
            # For each partition, for each fold, get the global test fold
            # from T and index the partition samples not in T as train set
            p_len = _partition(n_samples, n_partitions)

            p_last = 0
            for p_size in p_len:
                p_start, p_stop = p_last, p_last + p_size

                t_len = _partition(p_stop - p_start, n_splits)

                t_last = p_start
                for i, t_size in enumerate(t_len):
                    t_start, t_stop = t_last, t_last + t_size

                    # Get global test set indices
                    tei = T[i]

                    # Construct train set
                    tri_start_below, tri_stop_below = p_start, t_start
                    tri_start_above, tri_stop_above = t_stop, p_stop

                    tri = _prune_train(tri_start_below, tri_stop_below,
                                       tri_start_above, tri_stop_above)

                    yield tri, tei
                    t_last += t_size
                p_last += p_size


class FullIndex(BaseIndex):
    """Vacuous indexer to be used with final layers.

    FoldIndex is a compatibility class that stores the sample size to be
    predicted and yields a None, None index upon generation, although it
    is preferred to avoid calling FoldIndex for transparency.
    """
    def __init__(self, X=None, **kwargs):
        if X is not None:
            self.fit(X)

    def fit(self, X):
        """Store dimensionality data about X."""
        self.n_samples = X.shape[0]
        self.n_test_samples = X.shape[0]

    def _gen_indices(self):
        """Vacuous generator to ensure training data is not sliced."""
        yield None, None


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


def _check_subsample_index(n_samples, n_partitions, n_splits, raise_):
    """Check input validity of the SubsampleIndexer."""
    if not isinstance(n_partitions, Integral):
        raise ValueError("'n_partitions' must be an integer. "
                         "type(%s) was passed." % type(n_partitions))

    if not isinstance(n_splits, Integral):
        raise ValueError("'n_splits' must be an integer. "
                         "type(%s) was passed." % type(n_splits))

    if n_splits <= 1:
        if raise_ or n_partitions > 1:
            raise ValueError("Need at least 2 folds for splitting partitions. "
                             "Got %i." % n_splits)
        else:
            if n_partitions == 1 and n_splits == 1:
                warnings.warn("'n_splits' is 1, will return full index as "
                              "both training set and test set.")
    s = n_partitions * n_splits
    if s > n_samples:
        raise ValueError("Number of total splits %i is greater than the "
                         "number of samples: %i." % (s, n_samples))
