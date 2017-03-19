"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

estimation function for parallel preprocessing of the
:class:`mlens.ensemble.StackingEnsemble` class.
"""

from ..utils import safe_print, print_time, pickle_load, pickle_save
from ..utils.exceptions import FitFailedError, FitFailedWarning

from sklearn.base import clone

from joblib import delayed
import os

from time import time

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
        # (case-fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (est_name-fol_num, cloned_est)
        if kf is not None:
            fd = [('%s-%i' % (case, i),
                   tri,
                   tei,
                   [('%s-%i' % (n, i), clone(e)) for n, e in
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
        # Each est_list have entries (est_name-fol_num, cloned_est)
        if kf is not None:
            ls.extend([('%i' % i, tri, tei,
                        [('%s-%i' % (n, i), clone(e)) for n, e in
                         instance_list])
                       for i, (tri, tei) in enumerate(kf.split(X))
                       ])

    return ls


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
        # With cases, names are in the form (case_name-fold_num)
        # Otherwise named as (fold_num) - in this case the case name is None
        case = tup[0].split('-')[0] if '-' in tup[0] else None

        # Assign a column to estimators in belonging to the case-fold entry
        for est_name_fold_num, _ in tup[-1]:
            est_name = est_name_fold_num.split('-')[0]
            idx[tup[0], est_name_fold_num] = idx[case, est_name]

    return idx


def _assemble(temp_folder, instance_list, suffix):
    """Utility for loading fitted instances."""
    if suffix is 't':
        return [(case, pickle_load(os.path.join(temp_folder,
                                                '%s_%s' % (case, suffix))))
                for case, _, _, _ in instance_list]
    else:
        return [(case, pickle_load(os.path.join(temp_folder,
                                   '%s-%s_%s' % (case, est_name, suffix))))
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


def fit_trans(temp_folder, case, tr_list, X, y, idx, layer_name=None):
    """Fit transformers and write to cache."""
    xtrain = X[idx]
    ytrain = y[idx]
    out = []
    for tr_name, tr in tr_list:
        # Fit transformer
        _fit_tr(xtrain, ytrain, tr, tr_name, case, layer_name)

        # If more than one step, transform input for next step
        if len(tr_list) > 1:
            xtrain = _transform_tr(xtrain, tr, tr_name, case, layer_name)
        out.append((tr_name, tr))

    # Write transformer list to cache
    f = os.path.join(temp_folder, '%s_t' % case)
    pickle_save(out, f)


def fit_est(temp_folder, case, est_name, est, X, y, pred, idx,
            raise_on_exception=True, layer_name=None):
    """Fit estimator and write to cache along with predictions."""
    xtrain = X[idx[0]] if idx[0] is not None else X
    ytrain = y[idx[0]] if idx[0] is not None else y
    xtest = X[idx[1]] if idx[1] is not None else None

    # Load transformers
    f = os.path.join(temp_folder, '%s_t' % case)
    tr_list = pickle_load(f)

    # Transform input
    for tr_name, tr in tr_list:
        xtrain = _transform_tr(xtrain, tr, tr_name, case, layer_name)

        if xtest is not None:
            xtest = _transform_tr(xtest, tr, tr_name, case, layer_name)

    # Fit estimator
    est = _fit_est(xtrain, ytrain, est, raise_on_exception, est_name, case,
                   layer_name)

    # Predict if asked
    if xtest is not None:
        pred[idx[1], idx[2]] = _predict_est(xtest, est, raise_on_exception,
                                            est_name, case, layer_name)

    # We don't want to store the full test index:
    # during a process call, this index is a memmaped array, but once
    # the fitted ests are returned to the ensemble, the test indices would
    # be converted to full numpy arrays and get stored in the ensemble
    if idx[1] is not None:
        # Store as a tuple ((row_start, row_end + 1), col)
        # We add 1 to allow slicing (X[row_start:row_end + 1] == X[tei])
        idx = ((idx[1][0], idx[1][1] + 1), idx[2])
    else:
        idx = (None, idx[2])

    f = os.path.join(temp_folder, '%s-%s_e' % (case, est_name))
    pickle_save((est_name, est, idx), f)


def _fit(**kwargs):
    """Wrapper to select estimation or transformation."""
    if kwargs['name'] == 'transformation':
        fit_trans(**kwargs)
    else:
        fit_est(**kwargs)


def fit(layer, X, y, P, temp_folder, parallel, layer_name=None):
    """Fit :class:`layer` using the layer's ``indexer`` method.
    """
    if layer.verbose:
        printout = "stderr" if layer.verbose < 50 else "stdout"
        s = _name(layer_name, None)
        t0 = time()

    # Map transformers and estimators onto every fold
    estimator_folds = expand_instance_list(layer.estimators, layer.indexer, X)
    preprocessing_folds = expand_instance_list(layer.preprocessing,
                                               layer.indexer, X)

    # Get estimator prediction column mapping
    cm = get_col_idx(layer.preprocessing, layer.estimators, estimator_folds)

    # Fit preprocessing pipelines
    if layer.verbose:
        safe_print('\n%sFitting transformers' % s, file=printout)

    parallel(delayed(fit_trans)(temp_folder=temp_folder,
                                case=case,
                                name=None,
                                estimator=tr_list,
                                X=X,
                                y=y,
                                pred=None,
                                idx=tri,
                                layer_name=layer_name,
                                raise_on_exception=layer.raise_on_exception)
             for case, tri, _, tr_list in preprocessing_folds)

    # Fit estimators
    if layer.verbose:
        safe_print('\n%sFitting estimators' % s, file=printout)

    parallel(delayed(fit_est)(temp_folder=temp_folder,
                              case=case,
                              name=est_name,
                              estimator=est,
                              X=X,
                              y=y,
                              pred=P if tei is not None else None,
                              idx=(tri, tei, cm[case, est_name]),
                              layer_name=layer_name,
                              raise_on_exception=layer.raise_on_exception)
             for case, tri, tei, ests in estimator_folds
             for est_name, est in ests)

    # Assemble transformer list
    trans = _assemble(temp_folder, preprocessing_folds, 't')

    # Assemble estimator list
    ests = _assemble(temp_folder, estimator_folds, 'e')

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
    trans = dict(_strip(layer._layer_data['cases'], layer.preprocessing_))
    ests = _strip(layer._layer_data['cases'], layer.estimators_)

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
