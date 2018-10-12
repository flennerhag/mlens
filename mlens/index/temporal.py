"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Temporal (time series) indexing.
"""
from __future__ import division

from numbers import Integral
import numpy as np

from .base import BaseIndex
from ._checks import check_temporal_index


class TemporalIndex(BaseIndex):

    """Indexer that generates time series fold over ``X``.

    Sequential iterator that generates fold index tuples that preserve
    time series structure of data. Consequently, test folds always contain
    "future" observations (i.e. higher index values).

    The generator returns a tuple of stop and start positions to be used
    for numpy array slicing [stop:start].

    .. versionadded:: 0.2.3

    Parameters
    ----------
    step_size : int (default=1)
        number of samples to use in each test fold. The final window
        size may be smaller if too few observations remain.

    burn_in : int (default=None)
        number of samples to use for first training fold. These observations
        will be dropped from the output. Defaults to ``step_size``.

    window: int (default=None)
        number of previous samples to use in each training fold, except first
        which is determined by ``burn_in``. If ``None``, will use all previous
        observations.

    lag: int (default=0)
        distance between the most recent training point in the training fold and
        the first test point. For ``lag>0``, the training fold and the test fold
        will not be contiguous.

    X : array-like of shape [n_samples,] , optional
        the training set to partition. The training label array is also,
        accepted, as only the first dimension is used. If ``X`` is not
        passed at instantiating, the ``fit`` method must be called before
        ``generate``, or ``X`` must be passed as an argument of
        ``generate``.

    raise_on_exception : bool (default=True)
        whether to warn on suspicious slices or raise an error.

    See Also
    --------
    :class:`FoldIndex`, :class:`BlendIndex`, :class:`SubsetIndex`

    Examples
    --------

    >>> import numpy as np
    >>> from mlens.index import TemporalIndex
    >>> X = np.arange(10)
    >>> print("Data set: %r" % X)
    >>> print()
    >>>
    >>> idx = TemporalIndex(2, X=X)
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
    TRAIN IDX:                    array([0, 1]) | TEST IDX: array([2, 3, 4, 5, 6, 7, 8, 9])
    TRAIN IDX:              array([0, 1, 2, 3]) | TEST IDX: array([4, 5, 6, 7, 8, 9])
    TRAIN IDX:        array([0, 1, 2, 3, 4, 5]) | TEST IDX: array([6, 7, 8, 9])
    TRAIN IDX:  array([0, 1, 2, 3, 4, 5, 6, 7]) | TEST IDX:    array([8, 9])
    TRAIN SET:                    array([0, 1]) | TEST SET: array([2, 3, 4, 5, 6, 7, 8, 9])
    TRAIN SET:              array([0, 1, 2, 3]) | TEST SET: array([4, 5, 6, 7, 8, 9])
    TRAIN SET:        array([0, 1, 2, 3, 4, 5]) | TEST SET: array([6, 7, 8, 9])
    TRAIN SET:  array([0, 1, 2, 3, 4, 5, 6, 7]) | TEST SET:    array([8, 9])
    No overlap between train set and test set.
    """

    def __init__(self, step_size=1, burn_in=None, window=None, lag=0, X=None, raise_on_exception=True):
        super(TemporalIndex, self).__init__()
        self.step_size = step_size
        self.burn_in = burn_in if burn_in is not None else step_size
        self.window = window
        self.lag = lag
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
        check_temporal_index(
            self.burn_in, self.step_size, self.window,
            self.lag, self.n_samples, self.raise_on_exception)
        self.n_test_samples = self.n_samples - self.burn_in
        self.__fitted__ = True
        return self

    def _gen_indices(self):
        """Generate Time series folds"""
        idx = self.burn_in
        stop = False
        burn_in = True
        while not stop:

            train_stop = idx - self.lag
            if burn_in:
                train_start = 0
                burn_in = False
            elif self.window is None:
                train_start = 0
            else:
                train_start = max(idx - self.window - self.lag, 0)

            test_start = idx
            test_stop = min(idx + self.step_size, self.n_samples)

            train_index = (train_start, train_stop)
            test_index = (test_start, test_stop)

            yield train_index, test_index

            idx += self.step_size
            if idx >= self.n_samples:
                stop = True
