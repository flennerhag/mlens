"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

estimation function for parallel preprocessing of
:class:`mlens.ensemble.StackingEnsemble`.
"""

from numpy import asarray

from ..utils import safe_print, print_time, pickle_load, pickle_save
from ..utils.exceptions import FitFailedError, FitFailedWarning, \
    ParallelProcessingWarning, ParallelProcessingError

from sklearn.base import clone

from ..externals.joblib import delayed
import os

from time import time, sleep

import warnings


###############################################################################
def expand_instance_list(instance_list, kf=None, X=None):
    """Build a list of estimation tuples."""
    ls = list()

    if isinstance(instance_list, dict):
        # We need to build fit list on a case basis

        # --- Full data ---
        # Estimators to be fitted on full data. List entries have format:
        # (case, no_train_idx, no_test_idx, est_list)
        # Each est_list have entries (est_name, cloned_est)
        ls.extend([(case, None, None,
                    [(n, clone(e)) for n, e in instance_list[case]])
                   for case in sorted(instance_list)])

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (case__fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (est_name__fol_num, cloned_est)
        if kf is not None:
            fd = [('%s__%i' % (case, i),
                   (tri[0], tri[-1] + 1),
                   (tei[0], tei[-1] + 1),
                   [('%s__%i' % (n, i), clone(e)) for n, e in
                    instance_list[case]])
                  for case in sorted(instance_list)
                  for i, (tri, tei) in enumerate(kf.split(X))
                  ]
            ls.extend(fd)

    else:
        # No cases to worry about: expand the list of named instance tuples

        # --- Full data ---
        # Estimators to be fitted on full data. List entries have format:
        # (no_case, no_train_idx, no_test_idx, est_list)
        # Each est_list have entries (est_name, cloned_est)
        ls.extend([(None, None, None,
                    [(n, clone(e)) for n, e in instance_list])])

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (est_name__fol_num, cloned_est)
        if kf is not None:
            ls.extend([('%i' % i,
                        (tri[0], tri[-1] + 1),
                        (tei[0], tei[-1] + 1),
                        [('%s__%i' % (n, i), clone(e)) for n, e in
                         instance_list])
                       for i, (tri, tei) in enumerate(kf.split(X))
                       ])

    return ls


def _wrap(folded_list, name='__trans__'):
    """Wrap the folded transformer list.

    wraps the ``folded_transformer_list`` to give each entry the format
    ``(case, train_idx, None, [(__trans__, tr_list1), (__trans__, tr_list2)])``
    to be compatible with the ``folded_estimator_list`` in the call to
    ``parallel``."""
    return [(case, tri, None, [(name, instance_list)]) for
            case, tri, tei, instance_list in folded_list]


def get_col_idx(preprocessing, estimators, estimator_folds):
    """Utility for assigning each ``est`` in each ``prep`` a unique ``col_id``.

    Parameters
    ----------
    preprocessing : dict
        dictionary of preprocessing cases.

    estimators : dict
        dictionary of lists of estimators per preprocessing case.

    estimator_folds : list
        list of estimators per case and per cv fold

    """
    # Set up main columns mapping
    if isinstance(preprocessing, list) or preprocessing is None:
        idx = {(None, est_name): i for i, (est_name, _) in
               enumerate(estimators)}

        case_list = [None]
    else:
        # Nested for loop required
        case_list, idx, col = sorted(preprocessing), dict(), 0

        for case in case_list:
            for est_name, _ in estimators[case]:
                idx[case, est_name] = col
                col += 1

    # Map every estimator-case-fold entry back onto the just created column
    # mapping for estimators
    for tup in estimator_folds:
        if tup[0] in case_list:
            # A main estimator, will not be used for folded predictions
            continue

        # Get case name from the name_id entry
        # With cases, names are in the form (case_name__fold_num)
        # Otherwise named as (fold_num) - in this case the case name is None
        case = tup[0].split('__')[0] if '__' in tup[0] else None

        # Assign a column to estimators in belonging to the case__fold entry
        for est_name_fold_num, _ in tup[-1]:
            est_name = est_name_fold_num.split('__')[0]
            idx[tup[0], est_name_fold_num] = idx[case, est_name]

    return idx


def _assemble(dir, instance_list, suffix):
    """Utility for loading fitted instances."""
    if suffix is 't':
        return [(case, pickle_load(os.path.join(dir,
                                                '%s__%s' % (case, suffix))))
                for case, _, _, _ in instance_list]
    else:
        return [(case, pickle_load(os.path.join(dir,
                                   '%s__%s__%s' % (case, est_name, suffix))))
                for case, _, _, ests in instance_list
                for est_name, _ in ests]


def _strip(cases, fitted_estimators):
    """Strip all estimators not fitted on full data from list."""
    return [tup for tup in fitted_estimators if tup[0] in cases]


def _name(layer_name, case):
    """Utility for setting error or warning message prefix."""
    if layer_name is None and case is None:
        # Both empty
        out = ''
    elif layer_name is not None and case is not None:
        # Both full
        out = '[%s | %s ] ' % (layer_name, case)
    elif case is None:
        # Case empty, layer_name full
        out = '[%s] ' % layer_name
    else:
        # layer_name empty, case full
        out = '[%s] ' % case
    return out


###############################################################################
def _load_trans(f, case, lim, s, raise_on_exception):
    """Try loading transformers, and handle exception if not ready yet."""
    try:
        # Assume file exists
        return pickle_load(f)
    except FileNotFoundError or TypeError as exc:
        error_msg = ("The file %s cannot be found after %i seconds of waiting."
                     " Check that time to fit transformers is sufficiently "
                     "fast to complete fitting before fitting estimators. "
                     "Consider reducing the preprocessing intensity in the "
                     "ensemble, or increase the '__lim__' attribute to wait "
                     "extend period of waiting on transformation to complete. "
                     " Details:\n%r")

        if raise_on_exception:
            # Raise error immediately
            raise ParallelProcessingError(error_msg % exc)

        # Else, throw a warning and wait for transformation to finish
        warnings.warn("Could not find preprocessing case %s in the cache (%s)."
                      " Will check every %.1f seconds if pipeline has been "
                      "fitted for %i seconds before aborting. "
                      "Fitting ensembles with time consuming preprocessing "
                      "pipelines can cause estimators to call for "
                      "transformers that have not been fitted yet. "
                      "Consider optimizing preprocessing before passing to "
                      "ensemble. Details:\n%r" % (case, f, s, lim, exc),
                      ParallelProcessingWarning)

        ts = time()
        while not os.path.exists(f):
            sleep(s)
            if ts > lim:
                raise ParallelProcessingError(error_msg % exc)

    return pickle_load(f)


def _fit_tr(x, y, tr, tr_name, case, layer_name):
    """Wrapper around try-except block for fitting transformer."""
    try:
        tr.fit(x, y)
    except Exception as e:
        # Transformation is sequential: always throw error if one fails
        s = _name(layer_name, case)
        msg = "%sFitting transformer [%s] failed. Details:\n%r"
        raise FitFailedError(msg % (s, tr_name, e))


def _transform_tr(x, tr, tr_name, case, layer_name):
    """Wrapper around try-except block for transformer transformation."""
    try:
        return tr.transform(x)
    except Exception as e:
        s = _name(layer_name, case)
        msg = "%sTransformation with transformer [%s] of type (%s) failed. " \
              "Details:\n%r"
        raise FitFailedError(msg % (s, tr_name, tr.__class__, e))


def _fit_est(x, y, est, raise_on_exception, est_name, case, layer_name):
    """Wrapper around try-except block for estimator fitting."""
    try:
        return est.fit(x, y)
    except Exception as e:
        s = _name(layer_name, case)

        if raise_on_exception:
            raise FitFailedError("%sCould not fit estimator '%s'. "
                                 "Details:\n%r" % (s, est_name, e))

        msg = "%sCould not fit estimator '%s'. Will drop from " \
              "ensemble. Details:\n%r"
        warnings.warn(msg % (s, est_name, e), FitFailedWarning)


def _predict_est(x, est, raise_on_exception, est_name, case, layer_name):
    """Wrapper around try-except block for estimator predictions."""
    try:
        return est.predict(x)
    except Exception as e:
        s = _name(layer_name, case)

        if raise_on_exception:
            raise FitFailedError("%sCould not predict with estimator '%s'. "
                                 "Details:\n%r" % (s, est_name, e))

        msg = "%sCould not predict with estimator '%s'. Predictions will be" \
              "0. Details:\n%r"
        warnings.warn(msg % (s, est_name, e), FitFailedWarning)


def fit_trans(dir, case, inst, X, y, idx, layer_name=None):
    """Fit transformers and write to cache."""
    # Have to be careful in prepping data for estimation.
    # We need to slice memmap and convert to a proper array - otherwise
    # transformers can store results memmaped to the cache, which will
    # prevent the garbage collector from releasing the memmaps from memory
    # after estimation
    xtrain = asarray(X[idx[0]:idx[1]]) if idx is not None else asarray(X)
    ytrain = asarray(y[idx[0]:idx[1]]) if idx is not None else asarray(y)
    out = []
    for tr_name, tr in inst:
        # Fit transformer
        _fit_tr(xtrain, ytrain, tr, tr_name, case, layer_name)

        # If more than one step, transform input for next step
        if len(inst) > 1:
            xtrain = _transform_tr(xtrain, tr, tr_name, case, layer_name)
        out.append((tr_name, tr))

    # Write transformer list to cache
    f = os.path.join(dir, '%s__t' % case)
    pickle_save(out, f)


def fit_est(dir, case, name, inst, X, y, pred, idx,
            raise_on_exception=True, layer_name=None, lim=60, sec=0.1):
    """Fit estimator and write to cache along with predictions."""
    # Have to be careful in prepping data for estimation.
    # We need to slice memmap and convert to a proper array - otherwise
    # estimators can store results memmaped to the cache, which will
    # prevent the garbage collector from releasing the memmaps from memory
    # after estimation
    tri, tei, col = idx[0], idx[1], idx[2]

    x = asarray(X[tri[0]:tri[1]]) if tri is not None else asarray(X)
    ytrain = asarray(y[tri[0]:tri[1]]) if tri is not None else asarray(y)

    # Load transformers
    f = os.path.join(dir, '%s__t' % case)
    tr_list = _load_trans(f, case, lim, sec, raise_on_exception)

    # Transform input
    for tr_name, tr in tr_list:
        x = _transform_tr(x, tr, tr_name, case, layer_name)

    # Fit estimator
    est = _fit_est(x, ytrain, inst, raise_on_exception, name, case,
                   layer_name)

    # Predict if asked
    # The predict loop is kept separate to allow overwrite of x, thus keeping
    # only one subset of X in memory at any given time
    if tei is not None:
        x = asarray(X[tei[0]:tei[1]])

        for tr_name, tr in tr_list:
            x = _transform_tr(x, tr, tr_name, case, layer_name)

        pred[tei[0]:tei[1], col] = \
            _predict_est(x, est, raise_on_exception, name, case, layer_name)

    # We drop tri from index and only keep tei if any predictions were made
        idx = idx[1:]
    else:
        idx = (None, col)

    f = os.path.join(dir, '%s__%s__e' % (case, name))
    pickle_save((name, est, idx), f)


def _fit(**kwargs):
    """Wrapper to select fit_est or fit_trans."""
    f = fit_trans if kwargs['name'] == '__trans__' else fit_est
    f(**{k: v for k, v in kwargs.items() if k in f.__code__.co_varnames})


def fit(layer, X, y, P, dir, parallel, layer_name=None,
        lim=60, sec=0.1):
    """Fit :class:`layer` using the layer's ``indexer`` method."""
    if layer.verbose:
        printout = "stderr" if layer.verbose < 50 else "stdout"
        s = _name(layer_name, None)
        t0 = time()

    # Map transformers and estimators onto every fold
    est_folds = expand_instance_list(layer.estimators, layer.indexer, X)
    prep_folds = expand_instance_list(layer.preprocessing, layer.indexer, X)

    # Get estimator prediction column mapping
    cm = get_col_idx(layer.preprocessing, layer.estimators, est_folds)

    parallel(delayed(_fit)(dir=dir,
                           case=case,
                           name=name,
                           inst=instance,
                           X=X,
                           y=y,
                           pred=P if tei is not None else None,
                           idx=(tri, tei, cm[case, name])
                           if name != '__trans__' else tri,
                           layer_name=layer_name,
                           raise_on_exception=layer.raise_on_exception,
                           lim=lim, sec=sec)
             for case, tri, tei, instance_list in _wrap(prep_folds) + est_folds
             for name, instance in instance_list)

    # Assemble transformer list
    trans = _assemble(dir, prep_folds, 't')

    # Assemble estimator list
    ests = _assemble(dir, est_folds, 'e')

    if layer.verbose:
        print_time(t0, '%sDone' % s, file=printout)

    return ests, trans, None


###############################################################################
def _predict(case, tr_list, est_name, est, xtest, pred, col,
             raise_on_exception=True, layer_name=None):
    """Method for predicting with fitted transformers and estimators."""
    # Transform input
    for tr_name, tr in tr_list:
        xtest = _transform_tr(xtest, tr, tr_name, case, layer_name)

    # Predict into memmap
    pred[:, col] = _predict_est(xtest, est, raise_on_exception,
                                est_name, case, layer_name)


def predict_on_full(layer, X, P, parallel, layer_name=None):
    """Predict through :class:`layer` fitted through :func:`fit`."""
    if layer.verbose:
        printout = "stderr" if layer.verbose < 50 else "stdout"
        s = _name(layer_name, None)
        t0 = time()

    # Collect estimators fitted on full data
    trans = dict(_strip(layer.struct['cases'], layer.preprocessing_))
    ests = _strip(layer.struct['cases'], layer.estimators_)

    # Generate predictions
    if layer.verbose:
        safe_print('\n%sFitting estimators' % s, file=printout)

    parallel(delayed(_predict)(case=case,
                               tr_list=trans[case],
                               est_name=est_name,
                               est=est,
                               xtest=X,
                               pred=P,
                               col=col,
                               layer_name=layer_name,
                               raise_on_exception=layer.raise_on_exception)
             for case, (est_name, est, (_, col)) in ests)

    if layer.verbose:
        print_time(t0, '%sDone' % s, file=printout)
