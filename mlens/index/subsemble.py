"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Partitioning estimators
"""
from __future__ import division

import numpy as np

from ._checks import check_subsample_index
from .base import BaseIndex, partition, make_tuple, prune_train


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
    partitions : int, list (default = 2)
        Number of partitions to split data in. If ``partitions=1``,
        :class:`SubsetIndex` reduces to standard K-Fold.

    folds : int (default = 2)
        Number of splits to create in each partition. ``folds`` can
        not be 1 if ``n_partition > 1``. Note that if ``folds = 1``,
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
    >>> from mlens.index import SubsetIndex
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

    def __init__(self, partitions=2, folds=2, X=None, raise_on_exception=True):
        super(SubsetIndex, self).__init__()
        self.partitions = partitions
        self.folds = folds
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
        check_subsample_index(n, self.partitions, self.folds,
                              self.raise_on_exception)

        self.n_samples = self.n_test_samples = n
        self.__fitted__ = True
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
        if not self.__fitted__:
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
        parts = partition(self.n_samples, self.partitions)
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
        partitions = self.partitions
        n_samples = self.n_samples
        folds = self.folds

        # --- Create global test set folds ---
        # In each partition, the test set spans all partitions
        # Hence we must run through all partitions once first to
        # register
        # the global test set fold for each split of the folds

        # Partition sizes
        p_len = partition(n_samples, partitions)

        # Since the splitting is sequential and deterministic,
        # we build a
        # list of global test set indices. Hence, for any split, the
        # test index will be a tuple of indexes of test folds from each
        # partition. By concatenating these slices, the full test fold
        # is constructed.
        tei = [[] for _ in range(folds)]
        p_last = 0
        for p_size in p_len:
            p_start, p_stop = p_last, p_last + p_size

            t_len = partition(p_stop - p_start, folds)

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
        partitions = self.partitions
        n_samples = self.n_samples
        folds = self.folds

        T = self._build_test_sets()

        # For each partition, for each fold, get the global test fold
        # from T and index the partition samples not in T as train set
        p_len = partition(n_samples, partitions)

        p_last = 0
        for p_size in p_len:
            p_start, p_stop = p_last, p_last + p_size

            t_len = partition(p_stop - p_start, folds)

            t_last = p_start
            for i, t_size in enumerate(t_len):
                t_start, t_stop = t_last, t_last + t_size

                # Get global test set indices
                tei = T[i]

                # Construct train set
                tri_start_below, tri_stop_below = p_start, t_start
                tri_start_above, tri_stop_above = t_stop, p_stop

                tri = prune_train(tri_start_below, tri_stop_below,
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
    partition_estimator : instance
        Estimator to use for clustering.

    partitions : int
        Number of partitions the estimator will create.

    folds : int (default = 2)
        Number of folds to create in each partition. ``folds`` can
        not be 1 if ``n_partition > 1``. Note that if ``folds = 1``,
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
    >>> from mlens.index import ClusteredSubsetIndex
    >>>
    >>> km = KMeans(3, random_state=0)
    >>> X = np.arange(12).reshape(-1, 1); np.random.shuffle(X)
    >>> print("Data: {}".format(X.ravel()))
    >>>
    >>> s = ClusteredSubsetIndex(km)
    >>> s.fit(X)
    >>>
    >>> P = s.partition_estimator.predict(X)
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
                 partition_estimator,
                 partitions=2,
                 folds=2,
                 X=None,
                 y=None,
                 fit_estimator=True,
                 attr='predict',
                 partition_on='X',
                 raise_on_exception=True):
        super(ClusteredSubsetIndex, self).__init__()
        self.partition_estimator = partition_estimator
        self.fit_estimator = fit_estimator
        self.attr = attr
        self.partition_on = partition_on
        self.partitions = partitions
        self.folds = folds
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
                    self.partition_estimator.fit(X, y)
                except TypeError:
                    # Safeguard against estimators that do not accept y.
                    self.partition_estimator.fit(X)

            # Indexers are assumed to need fitting once, so we need to
            # generate cluster predictions during the fit call. To minimize
            # memory consumption, store cluster indexes as list of tuples
            self._clusters_ = self._get_partitions(X, y)
        self.__fitted__ = True
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

        f = getattr(self.partition_estimator, self.attr)
        if self.partition_on == 'X':
            cluster_ids = f(X)
        elif self.partition_on == 'y':
            cluster_ids = f(y)
        else:
            cluster_ids = f(X, y)

        clusters = np.unique(cluster_ids)
        self.partitions = len(clusters)

        # Condense the cluster index array into a list of tuples
        out = list()  # list of cluster indexes

        index = np.arange(n_samples)
        for c in clusters:
            cluster_index = index[cluster_ids == c]
            cluster_index_tup = make_tuple(cluster_index)
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
        folds = self.folds

        I = np.arange(n_samples)
        for prt in self._partition_generator(as_array=True):

            t_len = partition(prt.shape[0], folds)

            t_last = 0
            for t_size in t_len:
                t_start, t_stop = t_last, t_last + t_size

                tri = prt[t_start:t_stop]

                # Create test set by iterating over the index range
                tei = np.asarray([i for i in I if i not in tri])

                # Condense indexes to list of tuples
                tri = make_tuple(tri)
                tei = make_tuple(tei)

                yield tri, tei
                t_last += t_size
