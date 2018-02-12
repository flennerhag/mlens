"""ML-Ensemble

:author: Sebastian Flennerhag
:license: MIT
:copyright: 2017-2018

Functions for base computations
"""
from __future__ import division

import os
import warnings
from copy import deepcopy
from scipy.sparse import issparse
import numpy as np

from ..utils import pickle_load, pickle_save, load as _load
from ..utils.exceptions import MetricWarning, ParameterChangeWarning


def load(path, name, raise_on_exception=True):
    """Utility for loading from cache"""
    if isinstance(path, str):
        f = os.path.join(path, name)
        obj = _load(f, raise_on_exception)
    elif isinstance(path, list):
        obj = [tup[1] for tup in path if tup[0] == name]
        if not obj:
            raise ValueError(
                "No preprocessing pipeline in cache. Auxiliary Transformer "
                "have not cached pipelines, or cached to another sub-cache.")
        elif not len(obj) == 1:
            raise ValueError(
                "Could not load unique preprocessing pipeline. "
                "Transformer and/or Learner names are not unique")
        obj = obj[0]
    else:
        raise ValueError("Expected str or list. Got %r" % path)
    return obj


def save(path, name, obj):
    """Utility for saving to cache"""
    if isinstance(path, str):
        f = os.path.join(path, name)
        pickle_save(obj, f)
    elif isinstance(path, list):
        path.append((name, obj))


def prune_files(path, name):
    """Utility for safely selecting only relevant files"""
    if isinstance(path, str):
        files = [os.path.join(path, f)
                 for f in os.listdir(path)
                 if name == '.'.join(f.split('.')[:-3])]
        files = [pickle_load(f) for f in sorted(files)]
    elif isinstance(path, list):
        files = [tup[1] for tup in sorted(path, key=lambda x: x[0])
                 if name == '.'.join(tup[0].split('.')[:-2])]
    else:
        raise ValueError(
            "Expected name of cache or cache list. Got %r" % path)
    return files


def replace(source_files):
    """Utility function to replace empty files list"""
    replace_files = [deepcopy(o) for o in source_files]
    for o in replace_files:
        o.name = o.name[:-1] + '0'
        o.index = (o.index[0], 0)
        o.out_index = None
        o.in_index = None

    # Set a vacuous data list
    replace_data = [(o.name, None) for o in replace_files]
    return replace_files, replace_data


def mold_objects(learners, transformers):
    """Utility for enforcing compatible setup"""
    # TODO: remove
    out = []
    for objects in [learners, transformers]:
        if objects:
            if not isinstance(objects, list):
                objects = [objects]
        out.append(objects)
    return out


def set_output_columns(
        objects, n_partitions, multiplier, n_left_concats, target=None):
    """Set output columns on objects.

    Parameters
    ----------
    objects: list
        list of objects to set output columns on

    n_partitions: int
        number of partitions created by the indexer.

    multiplier: int
        number of columns claimed by each estimator.
        Typically 1, but can also be ``n_classes`` if
        making probability predictions

    n_left_concats: int
        number of columns to leave empty for left-concat

    target: int, optional
        target number of columns expected to be populated.
        Allows a check to ensure that all columns have been
        assigned.
    """
    col_index = n_left_concats
    col_map = list()
    sorted_learners = {obj.name:
                       obj for obj in objects}
    for _, obj in sorted(sorted_learners.items()):
        col_dict = dict()

        for partition_index in range(n_partitions):
            col_dict[partition_index] = col_index
            col_index += multiplier
        col_map.append([obj, col_dict])

    if (target) and (col_index != target):
        # Note that since col_index is incremented at the end,
        # the largest index_value we have col_index - 1
        raise ValueError(
            "Mismatch feature size in prediction array (%i) "
            "and max column index implied by learner "
            "predictions sizes (%i)" %
            (target, col_index - 1))

    for obj, col_dict in col_map:
        obj.output_columns = col_dict


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


def check_params(lpar, rpar):
    """Check parameter overlap

    Routine for checking two sets of parameter collections.
    :func:`check_params` iterate over items and expand nested parameter
    collections and test for equivalence of :class:`int`, :class:`float`,
    :class:`str` and :class:`bool` parameters.

    .. versionadded:: 0.2.0

    .. versionchanged:: 0.2.2
        Changed into a warning to prevent overly aggressive fails.

    Parameters
    ----------
    lpar : int, float, str, bool, iterable, estimator
        Default comparison set.

    rpar : int, float, str, bool, iterable, estimator
        Comparison set of fitted model.

    Returns
    -------
    pass : bool
        True if the two collections have equivalent parameter values, False
        otherwise.
    """
    # Expand estimator parameters
    if hasattr(lpar, 'get_params'):
        return check_params(lpar.get_params(deep=True),
                            rpar.get_params(deep=True))

    # Flatten dicts (also OrderedDicts)
    if isinstance(lpar, dict):
        par1, par2 = list(), list()
        for par in lpar:
            par1.append(lpar[par])
            par2.append(rpar[par])
        lpar, rpar = par1, par2

    # Iterate over flattened parameter collection
    if isinstance(lpar, (list, set, tuple)):
        for p1, p2 in zip(lpar, rpar):
            check_params(p1, p2)

    # --- param check ---
    _pass = True

    if (lpar is None) and not (rpar is None):
        _pass = False

    if isinstance(lpar, (str, bool)):
        _pass = lpar == rpar

    if isinstance(lpar, (int, float)):
        if np.isnan(lpar):
            _pass = np.isnan(rpar)
        elif np.isinf(lpar):
            _pass = np.isinf(rpar)
        else:
            _pass = lpar == rpar

    if not _pass:
        warnings.warn(
            "Parameter value (%r) has changed since model was fitted (%r)." %
            (lpar, rpar), ParameterChangeWarning)
    return _pass


def check_stack(new_items, stack):
    """Check if new items can safely be stacked onto old items"""
    names = [st.name for st in stack]
    for item in new_items:
        if item.name in names:
            raise ValueError("Name (%s) already exists in stack. "
                             "Rename before attempting to push." % item.name)
