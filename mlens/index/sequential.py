"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Sequential (time series) indexing.
"""
from __future__ import division

from numbers import Integral
import numpy as np

from .base import BaseIndex
from ._checks import check_sequential_index


class SequentialIndex(BaseIndex):

    """Indexer that generates time series fold over ``X``.

    Sequential iterator that generates fold index tuples that preserve
    time series structure of data. Consequently, test folds always contain
    "future" observations (i.e. higher index values).

    The generator returns a tuple of stop and start positions to be used
    for numpy array slicing [stop:start].

    .. versionadded:: 0.2.3

    Parameters
    ----------
    step_size : int
        number of samples to move fold window. Note that setting
        step_size = train_window will create non-overlapping training
        folds, while step_size < train_window will not.

    burn_in : int (default=step_size)
        number of samples to use for first training fold.

    train_window : int (default=None)
        number of samples to use in each training fold, except first which
        is determined by ``burn_in``. If ``None``, will use all previous
        observations.

    test_window : int (default=None)
        number of samples to use in each test fold. If ``None``,
        will use all remaining samples in the sequence. The final window
        size may be smaller if too few observations remain.

    X : array-like of shape [n_samples,] , optional
        the training set to partition. The training label array is also,
        accepted, as only the first dimension is used. If ``X`` is not
        passed at instantiating, the ``fit`` method must be called before
        ``generate``, or ``X`` must be passed as an argument of
        ``generate``.

    raise_on_exception : bool (default = True)
        whether to warn on suspicious slices or raise an error.

    See Also
    --------
    :class:`FoldIndex`, :class:`BlendIndex`, :class:`SubsetIndex`

    Examples
    --------

    >>> import numpy as np
    >>> from mlens.index import TimeSeriesIndex
    >>> X = np.arange(10)
    >>> print("Data set: %r" % X)
    >>> print()
    >>>
    >>> idx = TimeSeriesIndex(2, X)
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
    ...     assert max(train_idx) <= min(test_idx)
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
    """

    def __init__(self, step_size=1, burn_in=None,
                 train_window=None, test_window=None,
                 X=None, raise_on_exception=True):
        super(SequentialIndex, self).__init__()
        self.step_size = step_size
        self.burn_in = burn_in if burn_in is not None else step_size
        self.train_window = train_window
        self.test_window = test_window
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
        self.n_test_samples = self.n_samples = n
        check_sequential_index(
            self.burn_in, self.step_size, self.train_window,
            self.test_window, n, self.raise_on_exception)
        self.__fitted__ = True
        return self

    def _gen_indices(self):
        """Generate Time series folds"""
        idx = self.burn_in
        stop = False
        burn_in = True
        while not stop:

            train_stop = idx
            if burn_in:
                train_start = 0
                burn_in = False
            elif self.train_window is None:
                train_start = 0
            else:
                train_start = max(idx - self.train_window, 0)

            test_start = train_stop
            if self.test_window is None:
                test_stop = self.n_samples
            else:
                test_stop = min(idx + self.test_window, self.n_samples)

            train_index = (train_start, train_stop)
            test_index = (test_start, test_stop)

            yield train_index, test_index

            idx += self.step_size
            if idx >= self.n_samples:
                stop = True
