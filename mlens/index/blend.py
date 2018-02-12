"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Blend indexing.
"""
from __future__ import division

from numbers import Integral
import numpy as np

from ._checks import check_partial_index
from .base import BaseIndex


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
    >>> from mlens.index import BlendIndex
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
    >>> from mlens.index import BlendIndex
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
    >>> from mlens.index import BlendIndex
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
    >>> from mlens.index import BlendIndex
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
        super(BlendIndex, self).__init__()
        self.n_train = None
        self.n_test = None
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

        check_partial_index(self.n_samples, self.test_size, self.train_size,
                            self.n_test, self.n_train)

        self.n_test_samples = self.n_test

        self.__fitted__ = True
        return self

    def _gen_indices(self):
        """Return train and test set index generator."""
        # Blended train set is from 0 to n, with test set from n to N
        # There is no iteration.
        yield (0, self.n_train), (self.n_train, self.n_train + self.n_test)
