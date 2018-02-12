"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Class for identifying a training set after an estimator has been fitted. Used
for determining whether a `predict` or `transform` method should use
cross validation to create predictions, or estimators fitted on full
training data.
"""

from __future__ import division, print_function

from ..utils.exceptions import NotFittedError
from ..externals.sklearn.base import BaseEstimator

from numpy import array_equal, ix_
from numpy.random import permutation
from numbers import Integral


class IdTrain(BaseEstimator):

    """Container to identify training set.

    Samples a random subset from set passed to the `fit` method, to allow
    identification of the training set in a `transform` or `predict` method.

    Parameters
    ----------
    size : int
        size to sample. A random subset of size [size, size] will be stored
        in the instance.
    """

    def __init__(self, size=10):

        if not isinstance(size, Integral):
            raise ValueError("'size' must be an integer. Got %r" % size)

        self.size = size

    def fit(self, X):
        """Sample a training set.

        Parameters
        ----------
        X: array-like
            training set to sample observations from.

        Returns
        ----------
        self: obj
            fitted instance with stored sample.
        """
        self.train_shape = X.shape

        sample_idx = {}
        for i in range(2):
            dim_size = min(X.shape[i], self.size)
            sample_idx[i] = permutation(X.shape[i])[:dim_size]

        sample = X[ix_(sample_idx[0], sample_idx[1])]

        self.sample_idx_ = sample_idx
        self.sample_ = sample

        return self

    def is_train(self, X):
        """Check if an array is the training set.

        Parameters
        ----------
        X: array-like
            training set to sample observations from.

        Returns
        ----------
        self: obj
            fitted instance with stored sample.
        """
        if not hasattr(self, "train_shape"):
            raise NotFittedError("This IdTrain instance is not fitted yet.")

        if not self._check_shape(X):
            return False

        idx = self.sample_idx_

        try:
            # Grab sample from `X`
            sample = X[ix_(idx[0], idx[1])]

            return array_equal(sample, self.sample_)

        except IndexError:
            # If index is out of bounds, X.shape < training_set.shape
            # -> X is not the training set
            return False

    def _check_shape(self, X):
        """Check if X has the shape as the training set."""
        return all([X.shape[i] == self.train_shape[i] for i in range(2)])
