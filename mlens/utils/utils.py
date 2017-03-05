#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
"""

from __future__ import division, print_function, with_statement

from time import time
from pandas import DataFrame, Series
from numpy import array_equal
from numpy.random import permutation


try:
    import cPickle as pickle
except Exception:
    import pickle


def _slice(Z, idx):
    """Utility function for slicing either a DataFrame or an ndarray"""
    # determine training set
    if idx is None:
        z = Z.copy()
    else:
        if isinstance(Z, (DataFrame, Series)):
            z = Z.iloc[idx].copy()
        else:
            z = Z[idx].copy()
    return z


def pickle_save(obj, name):
    """Utility function for pickling an object"""
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(name):
    """Utility function for loading pickled object"""
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def print_time(t0, message='', **kwargs):
    """Utility function for printing time"""
    if len(message) > 0:
        message += ' | '

    r, s = divmod(time() - t0, 60)  # get sec
    h, m = divmod(r, 60)  # get h, min
    print(message + '%02d:%02d:%02d\n' % (h, m, s), **kwargs)


class IdTrain(object):

    """Container to identify training set

    Samples a random subset from set passed to the `fit` method, to allow
    identification of the training set in a `transform` or `predict` method.

    Parameters
    ----------
    size : int
        size to sample. A random subset of size [size, size] will be stored
        in the instance
    """

    def __init__(self, size=10):
        self.size = size

    def fit(self, X):
        """Sample a training set

        Parameters
        ----------
        X: array-like
            training set to sample observations from.

        Returns
        ----------
        self: obj
            fitted instance with stored sample
        """
        sample_idx = {}
        for i in range(2):
            dim_size = min(X.shape[i], self.size)
            sample_idx[i] = permutation(X.shape[i])[:dim_size]

        if isinstance(X, DataFrame):
            sample = X.iloc[sample_idx[0], sample_idx[1]]
        else:
            sample = X[sample_idx[0], sample_idx[1]]

        self.sample_idx_ = sample_idx
        self.sample_ = sample

        return self

    def is_train(self, X):
        """Check if an array is the training set

        Parameters
        ----------
        X: array-like
            training set to sample observations from.

        Returns
        ----------
        self: obj
            fitted instance with stored sample
        """
        idx = self.sample_idx_
        try:
            # Grab sample from `X`
            if isinstance(X, DataFrame):
                sample = X.iloc[idx[0], idx[1]].values
            else:
                sample = X[idx[0], idx[1]]
            return array_equal(sample, self.sample_)
        except IndexError:
            # X is not of the same shape as the training set > not training set
            return False
