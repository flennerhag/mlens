"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Stack indexing.
"""
from __future__ import division

from ._checks import check_full_index
from .base import BaseIndex


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
    >>> from mlens.index import FoldIndex
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

    Passing ``folds = 1`` without raising exception:

    >>> import numpy as np
    >>> from mlens.index import FoldIndex
    >>> X = np.arange(3)
    >>> print("Data set: %r" % X)
    >>> print()
    >>>
    >>> idx = FoldIndex(1, X, raise_on_exception=False)
    >>>
    >>> for train, test in idx.generate(as_array=True):
    ...     print('TRAIN IDX: %10r | TEST IDX: %10r' % (train, test))
    /../mlens/base/indexer.py:167: UserWarning: 'folds' is 1, will return
    full index as both training set and test set.
    warnings.warn("'folds' is 1, will return full index as "
    Data set: array([0, 1, 2])
    TRAIN IDX: array([0, 1, 2]) | TEST IDX: array([0, 1, 2])
    """

    def __init__(self, folds=2, X=None, raise_on_exception=True):
        super(FoldIndex, self).__init__()
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
        check_full_index(n, self.folds, self.raise_on_exception)

        self.n_test_samples = self.n_samples = n
        self.__fitted__ = True
        return self

    def _gen_indices(self):
        """Generate K-Fold iterator."""
        return super(FoldIndex, self)._gen_indices()
