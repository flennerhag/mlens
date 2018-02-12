"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT


Base classes for partitioning training data.
"""
from __future__ import division

from abc import abstractmethod
import numpy as np

from ..externals.sklearn.base import BaseEstimator


def prune_train(start_below, stop_below, start_above, stop_above):
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


def partition(n, p):
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

    >>> from mlens.index.base import partition
    >>> _partition(4, 2)
    array([2, 2])

    Return sample sizes of 3 partitions given a total of 8 samples

    >>> from mlens.index.base import partition
    >>> _partition(8, 3)
    array([3, 3, 2])
    """
    sizes = (n // p) * np.ones(p, dtype=np.int)
    sizes[:n % p] += 1
    return sizes


def make_tuple(arr):
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
    >>> from mlens.index.base import make_tuple
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


class BaseIndex(BaseEstimator):

    """Base Index class.

    Specification of indexer-wide methods and attributes that we can always
    expect to find in any indexer. Helps to provide a uniform interface
    during parallel estimation.
    """

    def __init__(self):
        self.folds = None
        self.partitions = 1
        self.n_samples = None
        self.n_test_samples = None

        self.__fitted__ = False

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
        n_samples = self.n_samples
        folds = self.folds

        if folds == 1:
            # Return the full index as both training and test set
            yield ((0, n_samples),), (0, n_samples)
        else:
            # Get the length of the test sets
            tei_len = partition(n_samples, folds)

            last = 0
            for size in tei_len:

                # Test set
                tei_start, tei_stop = last, last + size
                tei = (tei_start, tei_stop)

                # Train set
                tri_start_below, tri_stop_below = 0, tei_start
                tri_start_above, tri_stop_above = tei_stop, n_samples

                tri = prune_train(tri_start_below, tri_stop_below,
                                  tri_start_above, tri_stop_above)

                yield tri, tei
                last = tei_stop

    # pylint: disable=unused-argument, no-self-use
    def partition(self, X=None, as_array=False):
        """Partition generator method.

        Default behavior is to yield ``None``
        for fitting on full data. Overridden in
        :class:`SubsetIndex` and :class:`ClusteredSubsetIndex`
        to produce partition indexes.
        """
        yield None

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
        if not self.__fitted__:
            if X is None:
                raise AttributeError("No array provided to indexer. Either "
                                     "pass an array to the 'generate' method, "
                                     "or call the 'fit' method first or "
                                     "initiate the instance with an array X "
                                     "as argument.")
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

        >>> from mlens.index.base import BaseIndex
        >>> BaseIndex._build_range((0, 6))
        array([0, 1, 2, 3, 4, 5])

        Several slices (non-convex slicing)

        >>> from mlens.index.base import BaseIndex
        >>> BaseIndex._build_range([(0, 2), (4, 6)])
        array([0, 1, 4, 5])
        """
        if isinstance(idx[0], tuple):
            return np.hstack([np.arange(t0, t1) for t0, t1 in idx])
        return np.arange(idx[0], idx[1])

    def set_params(self, **params):
        self.__fitted__ = False
        return super(BaseIndex, self).set_params(**params)


class FullIndex(BaseIndex):

    """Vacuous indexer to be used with final layers.

    FullIndex is a compatibility class to be used with meta layers. It stores
    the sample size to be predicted for use with the
    :class:`ParallelProcessing` job manager, and yields a ``None, None``
    index when `generate` is called.
    """

    def __init__(self, X=None):
        super(FullIndex, self).__init__()
        if X is not None:
            self.fit(X)

    def fit(self, X, y=None, job=None):
        """Store dimensionality data about X."""
        self.n_samples = X.shape[0]
        self.n_test_samples = X.shape[0]
        self.__fitted__ = True

    def _gen_indices(self):
        """Vacuous generator to ensure training data is not sliced."""
        yield None, None
