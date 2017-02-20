#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 19/02/2017
licence: MIT
Suite for checking valid estimation and informative error traceback.
This is a light version of the Scikit-learn test suite, to pre-check
conditions for estimation before making estimator function calls in parallel
jobs.
"""
from sklearn.utils.validation import (check_random_state, check_X_y,
                                      check_array)


def check_inputs(X, y, random_state):
    """Permissive pre-check that an estimator is ready for fitting.

    Purpose is to enforce a minimum standard on X and y before passing
    subsets to estimators, as this will trigger excess copying in each
    worker during parallel estimation.

    Note that estimation may still fail or trigger excess copying if specific
    estimators have more restrictive input conditions.

    For full documentation, see `check_random_state` and `check_X_y` in
    `sklearn.utils.validation`.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.
    y : nd-array, list or sparse matrix
        Labels.

    Returns
    ---------
    X_converted : object
        The converted and validated X.
    y_converted : object
        The converted and validated y.
    """
    # Turn seed into a np.random.RandomState instance
    r = check_random_state(random_state) if random_state is not None else None

    # Check arrays
    if y is not None:
        X, y = \
            check_X_y(X, y,
                      accept_sparse=True,     # Sparse imput is admitted
                      dtype=None,             # Native dtype preserve
                      order=None,             # Make no C or Fortran imposition
                      copy=False,             # Do not trigger copying
                      force_all_finite=True,  # Raise error on np.inf or np.nan
                      ensure_2d=True,         # Force `X` do be a matrix
                      allow_nd=True,          # Allow `X.ndim` > 2
                      multi_output=True,      # Allow `y.shape[1]` > 1
                      ensure_min_samples=1,   # Do not impose minimum sample
                      ensure_min_features=1,  # Do not impose minimum features
                      warn_on_dtype=False     # Mute as `dtype` is `None`
                      )
    else:
        X = check_array(X,
                        accept_sparse=True,     # Sparse imput is admitted
                        dtype=None,             # Native dtype preserve
                        order=None,             # Make no C or Fortran impos
                        copy=False,             # Do not trigger copying
                        force_all_finite=True,  # Raise error on np.inf/np.nan
                        ensure_2d=True,         # Force `X` do be a matrix
                        allow_nd=True,          # Allow `X.ndim` > 2
                        ensure_min_samples=1,   # Do not impose minimum sample
                        ensure_min_features=1,  # Do not impose minimum feats
                        warn_on_dtype=False     # Mute as `dtype` is `None`
                        )
    return r


class NotFittedError(ValueError, AttributeError):
    """Error class for not fitted ensembles"""


def check_is_fitted(estimator, attr):
    """Check that ensemble has been fitted

    Parameters
    ----------
    ensemble : ensemble instance
        ensemble instance to check.
    attr : str
        attribute to assert existence of. Default is the 'layer_' attribute
        that holds fitted layers
    """
    msg = ("This %(name)s instance is not fitted yet. Call `fit` with "
           "appropriate arguments before using this method.")
    if not hasattr(estimator, attr):
        raise NotFittedError(msg % {'name': type(estimator).__name__})
