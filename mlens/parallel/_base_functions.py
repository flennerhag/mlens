"""
Functions for base computations
"""
from __future__ import division

import os
import warnings
from copy import deepcopy
from scipy.sparse import issparse
import numpy as np

from ..config import IVALS
from ..utils import pickle_load
from ..utils.exceptions import MetricWarning
from ..externals.joblib import delayed


def slice_array(x, y, idx, r=0):
    """Build training array index and slice data."""
    if idx == 'all':
        idx = None

    if idx:
        # Check if the idx is a tuple and if so, whether it can be made
        # into a simple slice
        if isinstance(idx[0], tuple):
            if len(idx[0]) > 1:
                # Advanced indexing is required. This will trigger a copy
                # of the slice in question to be made
                simple_slice = False
                idx = np.hstack([np.arange(t0 - r, t1 - r) for t0, t1 in idx])
                x = x[idx]
                y = y[idx] if y is not None else y
            else:
                # The tuple is of the form ((a, b),) and can be made
                # into a simple (a, b) tuple for which basic slicing applies
                # which allows a view to be returned instead of a copy
                simple_slice = True
                idx = idx[0]
        else:
            # Index tuples of the form (a, b) allows simple slicing
            simple_slice = True

        if simple_slice:
            x = x[slice(idx[0] - r, idx[1] - r)]
            y = y[slice(idx[0] - r, idx[1] - r)] if y is not None else y

    # Cast as ndarray to avoid passing memmaps to estimators
    if y is not None:
        y = y.view(type=np.ndarray)
    if not issparse(x):
        x = x.view(type=np.ndarray)

    return x, y


def assign_predictions(pred, p, tei, col, n):
    """Assign predictions to memmaped prediction array."""
    if tei == 'all':
        tei = None

    if tei is None:
        if len(p.shape) == 1:
            pred[:, col] = p
        else:
            pred[:, col:(col + p.shape[1])] = p
    else:
        r = n - pred.shape[0]

        if isinstance(tei[0], tuple):
            if len(tei) > 1:
                idx = np.hstack([np.arange(t0 - r, t1 - r) for t0, t1 in tei])
            else:
                tei = tei[0]
                idx = slice(tei[0] - r, tei[1] - r)
        else:
            idx = slice(tei[0] - r, tei[1] - r)

        if len(p.shape) == 1:
            pred[idx, col] = p
        else:
            pred[(idx, slice(col, col + p.shape[1]))] = p


def score_predictions(y, p, scorer, name, inst_name):
    """Try-Except wrapper around Learner scoring"""
    s = None
    if scorer is not None:
        try:
            s = scorer(y, p)
        except Exception as exc:
            warnings.warn("[%s] Could not score %s. Details:\n%r" %
                          (name, inst_name, exc), MetricWarning)
    return s


def transform(tr, x, y):
    """Try transforming with X and y. Else, transform with only X."""
    try:
        x = tr.transform(x)
    except TypeError:
        x, y = tr.transform(x, y)

    return x, y
