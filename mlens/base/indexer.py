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
    """Checks if indices above or below are empty and remove them.

    A utility function for checking if the train indices below the a given
    test set range are (0, 0), or if indices above the test set range is
    (n, n). In this case, these will lead to an empty array and therefore
    can safely be removed to create a single training set index range.

    Parameters
    ----------
    start_below : int
        index number starting below the test set. Should always be the same
        for all test sets.

     stop_below : int
        the index number at which the test set is starting on.

    start_above : int
    the index number at which the test set ends.

    stop_above : int
        The end of the data set (n). Should always be the same for all test
        sets.
    """
    if start_below == stop_below:
        tri = ((start_above, stop_above),)

    elif start_above == stop_above:
        tri = ((start_below, stop_below),)

    else:
        tri = ((start_below, stop_below), (start_above, stop_above))
    return tri


def _partition(n, p):
    """Get partition sizes for a given number of samples and partitions.

    This method will give an array containing the sizes of ``p`` partitions
    given a total sample size of ``n``. If there is a remainder from the
    split, the r first folds will be incremented by 1.

    Parameters
    ----------
    n : int
        number of samples.

    p : int
        number of partitions.

    Examples
    --------

    Return sample sizes of 2 partitions given a total of 4 samples

    >>> from mlens.base.indexer import _partition
    >>> _partition(4, 2)
    array([2, 2])

    Return sample sizes of 3 partitions given a total of 8 samples

    >>> from mlens.base.indexer import _partition
    >>> _partition(8, 3)
    array([3, 3, 2])
    """
    sizes = (n // p) * np.ones(p, dtype=np.int)
    sizes[:n % p] += 1
    return sizes

def _make_tuple(arr):
    """Make a list of index tuples from array

    Parameters
    ----------
    arr : array

    Returns
    -------
    out : list

    Examples
    --------
    >>> import numpy as np
    >>> from mlens.base.indexer import _make_tuple
    >>> _make_tuple(np.array([0, 1, 2, 5, 6, 8, 9, 10]))
    [(0, 3), (5, 7), (8, 11)]
    """
    out = list()
    t1 = t0 = arr[0]
    for i in arr[1:]:
        if i - t1 <= 1:
            t1 = i
            continue

        out.append((t0, t1 + 1))
        t1 = t0 = i

    out.append((t0, t1 + 1))
    return out


class BaseIndex(object):

    """Base Index class.

    Specification of indexer-wide methods and attributes that we can always
    expect to find in any indexer. Helps to provide a uniform interface
    during parallel estimation.
    """

    @abstractmethod
    def fit(self, X, y=None, job=None):
        """Method for storing array data.

        Parameters
        ----------
        X : array-like of shape [n_samples, optional]
            array to _collect dimension data from.

        y : array-like, optional
            label data

        job : str, optional
            optional job type data

        Returns
        -------
        instance :
            indexer with stores sample size data.

        Notes
        -----
        Fitting an indexer stores nothing that points to the array
        or memmap ``X``. Only the ``shape`` attribute of ``X`` is called.
        """

    @abstractmethod
    def _gen_indices(self):
        """Method for constructing the index generator.

        This should be modified by each indexer class to build the desired
        index. Currently, the Default is the standard K-Fold as this method
        is returned by Subset-based indexer when number of subsets is ``1``.

        Returns
        -------
        iterable :
            a generator of ``train_index, test_index``.
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
        r"""Front-end generator method.

        Generator for training and test set indices based on the
        generator specification in ``_gen_indicies``.

        Parameters
        ----------
        X : array-like, optional
            If instance has not been fitted, the training set ``X`` must be
            passed to the ``generate`` method, which will call ``fit`` before
            proceeding. If already fitted, ``X`` can be omitted.

        as_array : bool (default = False)
            whether to return train and test indices as a pair of tuple(s)
            or numpy arrays. If the returned tuples are singular they can be
            used on an array X with standard slicing syntax
            (``X[start:stop]``), but if a list of tuples is returned
            slicing ``X`` properly requires first building a list or array
            of index numbers from the list of tuples. This can be achieved
            either by setting ``as_array`` to ``True``, or running ::

                for train_tup, test_tup in indexer.generate():
                    train_idx = \
                        np.hstack([np.arange(t0, t1) for t0, t1 in train_tup])

            when slicing is required.
        """
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
        """Build an array of indexes from a list or tuple of index tuples.

        Given an index object containing tuples of ``(start, stop)`` indexes
        ``_build_range`` will return an array that concatenate all elements
        between each ``start`` and ``stop`` number.

        Examples
        --------
        Single slice (convex slicing)

        >>> from mlens.base.indexer import BaseIndex
        >>> BaseIndex._build_range((0, 6))
        array([0, 1, 2, 3, 4, 5])

        Several slices (non-convex slicing)

        >>> from mlens.base.indexer import BaseIndex
        >>> BaseIndex._build_range([(0, 2), (4, 6)])
        array([0, 1, 4, 5])
        """
        if isinstance(idx[0], tuple):
            return np.hstack([np.arange(t0, t1) for t0, t1 in idx])
        else:
            return np.arange(idx[0], idx[1])


class BlendIndex(BaseIndex):

    """Indexer that generates two non-overlapping subsets of ``X``.

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

    Parameters
    ----------
    test_size : int or float (default = 0.5)
        Size of the test set. If ``float``, assumed to be proportion of full
        data set.

    train_size : int or float, optional
        Size of test set. If not specified (i.e. ``train_size = None``,
        train_size is equal to ``n_samples - test_size``. If ``float``, assumed
        to be a proportion of full data set. If ``train_size`` + ``test_size``
        amount to less than the observations in the full data set, a subset
        of specified size will be used.

    X : array-like of shape [n_samples,] , optional
        the training set to partition. The training label array is also,
        accepted, as only the first dimension is used. If ``X`` is not
        passed
        at instantiation, the ``fit`` method must be called before
        ``generate``, or ``X`` must be passed as an argument of
        ``generate``.

    raise_on_exception : bool (default = True)
        whether to warn on suspicious slices or raise an error.

    See Also
    --------
    :class:`FoldIndex`, :class:`SubsetIndex`

    Examples
    --------

    Selecting an absolute test size, with train size as the remainder

    >>> import numpy as np
    >>> from mlens.base.indexer import BlendIndex
    >>> X = np.arange(8)
    >>> idx = BlendIndex(3, rebase=True)
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

    def __init__(self,
                 test_size=0.5,
                 train_size=None,
                 X=None,
                 raise_on_exception=True):

        self.test_size = test_size
        self.train_size = train_size
        self.raise_on_exception = raise_on_exception

        if X is not None:
            self.fit(X)

    def fit(self, X, y=None, job=None):
        """Method for storing array data.

        Parameters
        ----------
        X : array-like of shape [n_samples, optional]
            array to _collect dimension data from.
        y : None
            for compatibility
        job : None
            for compatibility

        Returns
        -------
        instance :
            indexer with stores sample size data.
        """
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
            self.n_train = int(np.floor(np.abs(self.n_samples - self.n_test)))

        elif isinstance(self.train_size, Integral):
            self.n_train = self.train_size

        else:
            self.n_train = int(np.floor(self.train_size * self.n_samples))

        _check_partial_index(self.n_samples, self.test_size, self.train_size,
                             self.n_test, self.n_train)

        self.n_test_samples = self.n_test

        return self

    def _gen_indices(self):
        """Return train and test set index generator."""
        # Blended train set is from 0 to n, with test set from n to N
        # There is no iteration.
        yield (0, self.n_train), (self.n_train, self.n_train + self.n_test)


class FoldIndex(BaseIndex):

    """Indexer that generates the full size of ``X``.

    K-Fold iterator that generates fold index tuples.

    FoldIndex creates a generator that returns a tuple of stop and start
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
    Simple clicing (i.e. ``X[start:stop]`` generally does not work for the
    train set, which often requires concatenating the train index range
    below the current test set, and the train index range above the current
    test set. To build get a training index, use ::

        ``hstack([np.arange(t0, t1) for t0, t1 in train_index_tuples])``.

    See Also
    --------
    :class:`BlendIndex`, :class:`SubsetIndex`

    Examples
    --------

    Creating arrays of folds and checking overlap

    >>> import numpy as np
    >>> from mlens.base.indexer import FoldIndex
    >>> X = np.arange(10)
    >>> print("Data set: %r" % X)
    >>> print()
    >>>
    >>> idx = FoldIndex(4, X)
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

    Passing ``n_splits = 1`` without raising exception.

    >>> import numpy as np
    >>> from mlens.base.indexer import FoldIndex
    >>> X = np.arange(3)
    >>> print("Data set: %r" % X)
    >>> print()
    >>>
    >>> idx = FoldIndex(1, X, raise_on_exception=False)
    >>>
    >>> for train, test in idx.generate(as_array=True):
    ...     print('TRAIN IDX: %10r | TEST IDX: %10r' % (train, test))
    /../mlens/base/indexer.py:167: UserWarning: 'n_splits' is 1, will return
    full index as both training set and test set.
    warnings.warn("'n_splits' is 1, will return full index as "

    Data set: array([0, 1, 2])
    TRAIN IDX: array([0, 1, 2]) | TEST IDX: array([0, 1, 2])
    """

    def __init__(self,
                 n_splits=2,
                 X=None,
                 raise_on_exception=True):

        self.n_splits = n_splits
        self.raise_on_exception = raise_on_exception

        if X is not None:
            self.fit(X)

    def fit(self, X, y=None, job=None):
        """Method for storing array data.

        Parameters
        ----------
        X : array-like of shape [n_samples, optional]
            array to _collect dimension data from.
        y : None
            for compatibility
        job : None
            for compatibility

        Returns
        -------
        instance :
            indexer with stores sample size data.
        """
        n = X.shape[0]
        _check_full_index(n, self.n_splits, self.raise_on_exception)

        self.n_test_samples = self.n_samples = n

        return self

    def _gen_indices(self):
        """Generate K-Fold iterator."""
        return super(FoldIndex, self)._gen_indices()


class SubsetIndex(BaseIndex):

    r"""Subsample index generator.

    Generates cross-validation folds according used to create ``J``
    partitions of the data and ``v`` folds on each partition according to as
    per [#]_:

        1. Split ``X`` into ``J`` partitions

        2. For each partition:

            (a) For each fold ``v``, create train index of all idx not in ``v``

            (b) Concatenate all the fold ``v`` indices into a test index for
                fold ``v`` that spans all partitions

    Setting ``J = 1`` is equivalent to the :class:`FullIndexer`, which returns
    standard K-Fold train and test set indices.

    See Also
    --------
    :class:`FoldIndex`, :class:`BlendIndex`, :class:`Subsemble`

    References
    ----------
    .. [#] Sapp, S., van der Laan, M. J., & Canny, J. (2014). Subsemble: an
       ensemble method for combining subset-specific algorithm fits. Journal
       of Applied Statistics, 41(6), 1247-1259.
       http://doi.org/10.1080/02664763.2013.864263

    Parameters
    ----------
    n_partitions : int, list (default = 2)
        Number of partitions to split data in. If ``n_partitions=1``,
        :class:`SubsetIndex` reduces to standard K-Fold.

    n_splits : int (default = 2)
        Number of splits to create in each partition. ``n_splits`` can
        not be 1 if ``n_partition > 1``. Note that if ``n_splits = 1``,
        both the train and test set will index the full data.

    X : array-like of shape [n_samples,] , optional
        the training set to partition. The training label array is also,
        accepted, as only the first dimension is used. If ``X`` is not
        passed at instantiation, the ``fit`` method must be called before
        ``generate``, or ``X`` must be passed as an argument of
        ``generate``.

    raise_on_exception : bool (default = True)
        whether to warn on suspicious slices or raise an error.

    Examples
    --------
    >>> import numpy as np
    >>> from mlens.base import SubsetIndex
    >>> X = np.arange(10)
    >>> idx = SubsetIndex(3, X=X)
    >>>
    >>> print('Expected partitions of X:')
    >>> print('J = 1: {!r}'.format(X[0:4]))
    >>> print('J = 2: {!r}'.format(X[4:7]))
    >>> print('J = 3: {!r}'.format(X[7:10]))
    >>> print('SubsetIndexer partitions:')
    >>> for i, part in enumerate(idx.partition(as_array=True)):
    ...     print('J = {}: {!r}'.format(i + 1, part))
    >>> print('SubsetIndexer folds on partitions:')
    >>> for i, (tri, tei) in enumerate(idx.generate()):
    ...     fold = i % 2 + 1
    ...     part = i // 2 + 1
    ...     train = np.hstack([np.arange(t0, t1) for t0, t1 in tri])
    ...     test = np.hstack([np.arange(t0, t1) for t0, t1 in tei])
    >>>     print("J = %i | f = %i | "
    ...           "train: %15r | test: %r" % (part, fold, train, test))
    Expected partitions of X:
    J = 1: array([0, 1, 2, 3])
    J = 2: array([4, 5, 6])
    J = 3: array([7, 8, 9])
    SubsetIndexer partitions:
    J = 1: array([0, 1, 2, 3])
    J = 2: array([4, 5, 6])
    J = 3: array([7, 8, 9])
    SubsetIndexer folds on partitions:
    J = 1 | f = 1 | train:   array([2, 3]) | test: array([0, 1, 4, 5, 7, 8])
    J = 1 | f = 2 | train:   array([0, 1]) | test: array([2, 3, 6, 9])
    J = 2 | f = 1 | train:      array([6]) | test: array([0, 1, 4, 5, 7, 8])
    J = 2 | f = 2 | train:   array([4, 5]) | test: array([2, 3, 6, 9])
    J = 3 | f = 1 | train:      array([9]) | test: array([0, 1, 4, 5, 7, 8])
    J = 3 | f = 2 | train:   array([7, 8]) | test: array([2, 3, 6, 9])
    """

    def __init__(self,
                 n_partitions=2,
                 n_splits=2,
                 X=None,
                 raise_on_exception=True):

        self.n_partitions = n_partitions
        self.n_splits = n_splits
        self.raise_on_exception = raise_on_exception

        if X is not None:
            self.fit(X)

    def fit(self, X, y=None, job=None):
        """Method for storing array data.

        Parameters
        ----------
        X : array-like of shape [n_samples, optional]
            array to _collect dimension data from.
        y : None
            for compatibility
        job : None
            for compatibility

        Returns
        -------
        instance :
            indexer with stores sample size data.
        """
        n = X.shape[0]
        _check_subsample_index(n, self.n_partitions, self.n_splits,
                               self.raise_on_exception)

        self.n_samples = self.n_test_samples = n
        return self

    def partition(self, X=None, as_array=False):
        """Get partition indices for training full subset estimators.

        Returns the index range for each partition of X.

        Parameters
        ----------
        X : array-like of shape [n_samples,] , optional
            the training set to partition. The training label array is also,
            accepted, as only the first dimension is used. If ``X`` is not
            passed at instantiation, the ``fit`` method must be called before
            ``generate``, or ``X`` must be passed as an argument of
            ``generate``.

        as_array : bool (default = False)
            whether to return partition as an index array. Otherwise tuples
            of ``(start, stop)`` indices are returned.
        """
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

        # Return the partition indices.
        parts = _partition(self.n_samples, self.n_partitions)
        last = 0
        for size in parts:
            idx = last, last + size

            if as_array:
                idx = self._build_range(idx)

            yield idx

            last += size

    def _build_test_sets(self):
        """Build global test folds for each split of every partition.

        This method runs through each partition and fold to register all the
        test set indices across partitions. For each test fold ``i``, the test
        set indices are thus the union of fold ``i`` indices across all ``J``
        partitions.
        """
        n_partitions = self.n_partitions
        n_samples = self.n_samples
        n_splits = self.n_splits

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
        """Create generator for subsample.

        Generate indices of training set and test set for
            - each partition
            - each fold in the partition

        Note that the test index return is *global*, i.e. it contains the
        test indices of that fold across partitions. See Examples for
        further details.
        """
        n_partitions = self.n_partitions
        n_samples = self.n_samples
        n_splits = self.n_splits

        T = self._build_test_sets()

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


class ClusteredSubsetIndex(BaseIndex):

    """Clustered Subsample index generator.

    Generates cross-validation folds according used to create ``J``
    partitions of the data and ``v`` folds on each partition according to as
    per [#]_:

        1. Split ``X`` into ``J`` partitions

        2. For each partition:

            (a) For each fold ``v``, create train index of all idx not in ``v``

            (b) Concatenate all the fold ``v`` indices into a test index for
                fold ``v`` that spans all partitions


    Setting ``J = 1`` is equivalent to the :class:`FullIndexer`, which returns
    standard K-Fold train and test set indices.

    :class:`ClusteredSubsetIndex` uses a user-provided estimator to partition
    the data, in contrast to the :class:`SubsetIndex` generator, which
    partitions data into randomly into equal sizes.

    See Also
    --------
    :class:`FoldIndex`, :class:`BlendIndex`, :class:`SubsetIndex`

    References
    ----------
    .. [#] Sapp, S., van der Laan, M. J., & Canny, J. (2014). Subsemble: an
       ensemble method for combining subset-specific algorithm fits. Journal
       of Applied Statistics, 41(6), 1247-1259.
       http://doi.org/10.1080/02664763.2013.864263

    Parameters
    ----------
    estimator : instance
        Estimator to use for clustering.

    n_partitions : int
        Number of partitions the estimator will create.

    n_splits : int (default = 2)
        Number of folds to create in each partition. ``n_splits`` can
        not be 1 if ``n_partition > 1``. Note that if ``n_splits = 1``,
        both the train and test set will index the full data.

    fit_estimator : bool (default = True)
        whether to fit the estimator separately before generating labels.

    attr : str (default = 'predict')
        the attribute to use for generating cluster membership labels.

    X : array-like of shape [n_samples,] , optional
        the training set to partition. The training label array is also,
        accepted, as only the first dimension is used. If ``X`` is not
        passed at instantiation, the ``fit`` method must be called before
        ``generate``, or ``X`` must be passed as an argument of
        ``generate``.

    raise_on_exception : bool (default = True)
        whether to warn on suspicious slices or raise an error.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> from mlens.base.indexer import ClusteredSubsetIndex
    >>>
    >>> km = KMeans(3, random_state=0)
    >>> X = np.arange(12).reshape(-1, 1); np.random.shuffle(X)
    >>> print("Data: {}".format(X.ravel()))
    >>>
    >>> s = ClusteredSubsetIndex(km)
    >>> s.fit(X)
    >>>
    >>> P = s.estimator.predict(X)
    >>> print("cluster labels: {}".format(P))
    >>>
    >>> for j, i in enumerate(s.partition(as_array=True)):
    ...    print("partition ({}) index: {}, cluster labels: {}".format(i, j + 1, P[i]))
    >>>
    >>> for i in s.generate(as_array=True):
    ...     print("train fold index: {}, cluster labels: {}".format(i[0], P[i[0]]))
    Data: [ 8  7  5  2  4 10 11  1  3  6  9  0]
    cluster labels: [0 2 2 1 2 0 0 1 1 2 0 1]
    partition (1) index: [ 0  5  6 10], cluster labels: [0 0 0 0]
    partition (2) index: [ 3  7  8 11], cluster labels: [1 1 1 1]
    partition (3) index: [1 2 4 9], cluster labels: [2 2 2 2]
    train fold index: [0 3 5], cluster labels: [0 0 0]
    train fold index: [ 6 10], cluster labels: [0 0]
    train fold index: [2 7], cluster labels: [1 1]
    train fold index: [ 9 11], cluster labels: [1 1]
    train fold index: [1 4], cluster labels: [2 2]
    train fold index: [8], cluster labels: [2]
    """

    def __init__(self,
                 estimator,
                 n_partitions=2,
                 n_splits=2,
                 X=None,
                 y=None,
                 fit_estimator=True,
                 attr='predict',
                 partition_on='X',
                 raise_on_exception=True):

        self.estimator = estimator
        self.fit_estimator = fit_estimator
        self.attr = attr
        self.partition_on = partition_on
        self.n_partitions = n_partitions
        self.n_splits = n_splits
        self.raise_on_exception = raise_on_exception

        self._clusters_ = None
        if X is not None:
            self.fit(X, y)

    def fit(self, X, y=None, job='fit'):
        """Method for storing array data.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            input array.

        y : array-like of shape [n_samples, ]
            labels.

        job : str, ['fit', 'predict'] (default='fit')
            type of estimation job. If 'fit', the indexer will be fitted,
            which involves fitting the estimator. Otherwise, the indexer will
            not be fitted (since it is not used for prediction).

        Returns
        -------
        instance :
            indexer with stores sample size data.
        """
        n = X.shape[0]
        self.n_samples = self.n_test_samples = n

        if 'fit' in job:
            # Only generate new clusters if fitting an ensemble
            if self.fit_estimator:
                try:
                    self.estimator.fit(X, y)
                except TypeError:
                    # Safeguard against estimators that do not accept y.
                    self.estimator.fit(X)

            # Indexers are assumed to need fitting once, so we need to
            # generate cluster predictions during the fit call. To minimize
            # memory consumption, store cluster indexes as list of tuples
            self._clusters_ = self._get_partitions(X, y)

        return self

    def partition(self, X=None, y=None, as_array=False):
        """Get partition indices for training full subset estimators.

        Returns the index range for each partition of X.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features] , optional
            the set to partition. The training label array is also,
            accepted, as only the first dimension is used. If ``X`` is not
            passed at instantiation, the ``fit`` method must be called before
            ``generate``, or ``X`` must be passed as an argument of
            ``generate``.

        y : array-like of shape [n_samples,], optional
            the labels of the set to partition.

        as_array : bool (default = False)
            whether to return partition as an index array. Otherwise tuples
            of ``(start, stop)`` indices are returned.
        """
        if X is not None:
            self.fit(X, y)
        return self._partition_generator(as_array)

    def _partition_generator(self, as_array):
        """Generator for partitions.

        Parameters
        ----------
        as_array : bool:
            whether to return partition indexes as a list of index tuples, or
            as an array.
        """
        for cluster_index in self._clusters_:
            if as_array:
                yield self._build_range(cluster_index)
            else:
                yield cluster_index

    def _get_partitions(self, X, y=None):
        """Get clustered partition indices from estimator.

        Returns the index range for each partition of X. See :func:`partition`
        for further details.
        """
        n_samples = X.shape[0]

        f = getattr(self.estimator, self.attr)
        if self.partition_on == 'X':
            cluster_ids = f(X)
        elif self.partition_on == 'y':
            cluster_ids = f(y)
        else:
            cluster_ids = f(X, y)

        clusters = np.unique(cluster_ids)
        self.n_partitions = len(clusters)

        # Condense the cluster index array into a list of tuples
        out = list()  # list of cluster indexes

        index = np.arange(n_samples)
        for c in clusters:
            cluster_index = index[cluster_ids == c]
            cluster_index_tup = _make_tuple(cluster_index)
            out.append(cluster_index_tup)
        return out

    def _gen_indices(self):
        """Generator for clustered subsample.

        Generate indices of training set and test set for
            - each partition
            - each fold in the partition

        Note that the test index return is *global*, i.e. it contains the
        test indices of that fold across partitions.
        """
        n_samples = self.n_samples
        n_splits = self.n_splits

        I = np.arange(n_samples)
        for partition in self._partition_generator(as_array=True):

            t_len = _partition(partition.shape[0], n_splits)

            t_last = 0
            for i, t_size in enumerate(t_len):
                t_start, t_stop = t_last, t_last + t_size

                tri = partition[t_start:t_stop]

                # Create test set by iterating over the index range
                tei = np.asarray([i for i in I if i not in tri])

                # Condense indexes to list of tuples
                tri = _make_tuple(tri)
                tei = _make_tuple(tei)

                yield tri, tei
                t_last += t_size


class FullIndex(BaseIndex):

    """Vacuous indexer to be used with final layers.

    FullIndex is a compatibility class to be used with meta layers. It stores
    the sample size to be predicted for use with the
    :class:`ParallelProcessing` job manager, and yields a ``None, None``
    index when `generate` is called. However, it is preferable to build code
    that avoids call the ``generate`` method when the indexer is known to be
    an instance of FullIndex for transparency and maintainability.
    """

    def __init__(self, X=None):
        if X is not None:
            self.fit(X)

    def fit(self, X, y=None, job=None):
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


def _check_partial_index(n_samples, test_size, train_size, n_test, n_train):
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

    if not n_partitions > 0:
        raise ValueError("'n_partitions' must be a positive integer. "
                         "{} was passed.".format(n_partitions))

    if not isinstance(n_splits, Integral):
        raise ValueError("'n_splits' must be an integer. "
                         "type(%s) was passed." % type(n_splits))

    if n_splits == 1:
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
