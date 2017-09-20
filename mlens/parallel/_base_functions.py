"""
Functions for base computations
"""

import os
import numpy as np
from time import sleep
from scipy.sparse import issparse
from copy import deepcopy

from ..externals.joblib import delayed
from ..utils import pickle_load, pickle_save
from ..utils.exceptions import (ParallelProcessingError,
                                ParallelProcessingWarning)

try:
    from time import perf_counter as time_
except ImportError:
    from time import time as time_

import warnings


# Default params
IVALS = (0.01, 120)
def predict_fold_est(*args): pass
def fit_est(*args): pass
def predict(*args): pass
def fit_trans(*args): pass

###############################################################################
# Base estimation jobs

def fit(inst, X, y, P, dir, parallel):
    """Fit layer through given attribute."""
    # Set estimator and transformer lists to loop over, and collect
    # estimator column ids for the prediction matrix
    inst._format_instance_list()
    inst._get_col_id()

    # Auxiliary variables
    preprocess = inst.t is not None
    pred_method = inst.layer._predict_attr

    if preprocess:
        parallel(delayed(fit_trans)(dir=dir,
                                    case=case,
                                    inst=instance_list,
                                    x=X,
                                    y=y,
                                    idx=tri)
                 for case, tri, _, instance_list in inst.t)

    parallel(delayed(fit_est)(dir=dir,
                              case=case,
                              inst_name=inst_name,
                              inst=instance,
                              x=X,
                              y=y,
                              pred=P if tei is not None else None,
                              idx=(tri, tei, inst.c[case, inst_name]),
                              name=inst.layer.name,
                              raise_on_exception=inst.layer.raise_on_exception,
                              preprocess=preprocess,
                              ivals=IVALS,
                              attr=pred_method,
                              scorer=inst.layer.scorer)
             for case, tri, tei, instance_list in inst.e
             for inst_name, instance in instance_list)
    assemble(inst)


def predict(inst, X, P, parallel):
    """Predict full X array using fitted layer."""
    inst._check_fitted()
    prep, ests = inst._retrieve('full')

    parallel(delayed(predict_est)(tr_list=deepcopy(prep[case])
                                  if prep is not None else [],
                                  est=est,
                                  x=X,
                                  pred=P,
                                  col=col,
                                  attr=inst.layer._predict_attr)
             for case, (_, est, (_, col)) in ests)


def transform(inst, X, P, parallel):
    """Transform training data with fold-estimators, as in ``fit`` call."""
    inst._check_fitted()
    prep, ests = inst._retrieve('fold')

    parallel(delayed(predict_fold_est)(tr_list=deepcopy(prep[case])
                                       if prep is not None else [],
                                       est=est,
                                       x=X,
                                       pred=P,
                                       idx=idx,
                                       attr=inst.layer._predict_attr)
             for case, (est_name, est, idx) in ests)

###############################################################################
# Helpers
def assemble(inst):
    """Store fitted transformer and estimators in the layer."""
    inst.layer.preprocessing_ = _assemble(inst.dir, inst.t, 't')
    inst.layer.estimators_, s = _assemble(inst.dir, inst.e, 'e')

    if inst.layer.scorer is not None and inst.layer.cls is not 'full':
        inst.layer.scores_ = inst._build_scores(s)


def construct_args(func, job):
    """Helper to construct argument list from a ``job`` instance."""
    fargs = func.__code__.co_varnames

    # Strip undesired variables
    args = [a for a in fargs if a not in {'parallel', 'X', 'P', 'self'}]

    kwargs = {a: getattr(job, a) for a in args if a in job.__slots__}

    if 'X' in fargs:
        kwargs['X'] = job.predict_in
    if 'P' in fargs:
        kwargs['P'] = job.predict_out
    return kwargs


def _slice_array(x, y, idx, r=0):
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


def _assign_predictions(pred, p, tei, col, n):
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


def _score_predictions(y, p, scorer, name, inst_name):
    s = None
    if scorer is not None:
        try:
            s = scorer(y, p)
        except Exception as exc:
            warnings.warn("[%s] Could not score %s. Details:\n%r" %
                          (name, inst_name, exc),
                          ParallelProcessingWarning)
    return s


def _assemble(dir, instance_list, suffix):
    """Utility for loading fitted instances."""
    if suffix is 't':
        if instance_list is None:
            return

        return [(tup[0],
                 pickle_load(os.path.join(dir, '%s__%s' % (tup[0], suffix))))
                for tup in instance_list]
    else:
        # We iterate over estimators to split out the estimator info and the
        # scoring info (if any)
        ests_ = []
        scores_ = []
        for tup in instance_list:
            for etup in tup[-1]:
                f = os.path.join(dir, '%s__%s__%s' % (tup[0], etup[0], suffix))
                loaded = pickle_load(f)

                # split out the scores, the final element in the l tuple
                ests_.append((tup[0], loaded[:-1]))

                case = '%s___' % tup[0] if tup[0] is not None else '___'
                scores_.append((case + etup[0], loaded[-1]))

        return ests_, scores_


def _transform(tr, x, y):
    """Try transforming with X and y. Else, transform with only X."""
    try:
        x = tr.transform(x)
    except TypeError:
        x, y = tr.transform(x, y)

    return x, y


def _load_trans(dir, ivals, raise_on_exception):
    """Try loading transformers, and handle exception if not ready yet."""
    s = ivals[0]
    lim = ivals[1]
    try:
        return pickle_load(dir)
    except (OSError, IOError) as exc:
        msg = str(exc)

        ts = time_()
        while not os.path.exists(dir):
            sleep(s)

            if time_() - ts > lim:
                if raise_on_exception:
                    raise ParallelProcessingError(
                        "Could not load transformer at %s\nDetails:\n%r" %
                        (dir, msg))

                warnings.warn("Could not load transformer at %s. "
                              "Will check every %.1f seconds for %i seconds "
                              "before aborting. " % (dir, s, lim),
                              ParallelProcessingWarning)

                # Set raise_on_exception to True now to ensure timeout
                raise_on_exception = True
                ts = time_()

        return pickle_load(dir)
