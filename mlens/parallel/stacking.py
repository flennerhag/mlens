"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Estimation engine for parallel preprocessing of stacked layer.
"""

from numpy import asarray, hstack, arange

from ..utils import (safe_print, print_time, pickle_load, pickle_save,
                     check_is_fitted)
from ..utils.exceptions import (FitFailedError, FitFailedWarning,
                                NotFittedError, PredictFailedError,
                                PredictFailedWarning,
                                ParallelProcessingWarning,
                                ParallelProcessingError)

from sklearn.base import clone

from joblib import delayed
import os

from time import time, sleep

import warnings


###############################################################################
class Stacker(object):
    """Stacked fit sub-process class.

    Class for fitting a Layer using Stacking or Blending.
    """

    __slots__ = ['verbose', 'layer', 'raise_', 'name',
                 'lim', 'ival', 'dual', 'e', 'p', 'c']

    def __init__(self, layer, dual=True):
        self.layer = layer

        # Copy some layer parameters to ease notation
        self.verbose = self.layer.verbose
        self.raise_ = self.layer.raise_on_exception
        self.name = self.layer.name
        self.lim = getattr(layer, 'lim', 600)
        self.ival = getattr(layer, 'ival', 0.1)

        self._prepare_estimators()
        self._get_col_id()

        self.dual = dual

    def __call__(self, attr, *args, **kwargs):
        """Generic argument agnostic call function to a Stacker method."""
        getattr(self, attr)(*args, **kwargs)

    def _prepare_estimators(self):
        """Expand estimator and preprocessing lists to include folds."""
        self.e = expand_instance_list(self.layer.estimators,
                                      self.layer.indexer)

        self.p = expand_instance_list(self.layer.preprocessing,
                                      self.layer.indexer)

    def _get_col_id(self):
        """Assign unique col_id to every estimator."""
        self.c = get_col_idx(self.layer.preprocessing,
                             self.layer.estimators,
                             self.e)

    def fit(self, X, y, P, dir, parallel):
        """Fit layer."""
        if self.verbose:
            printout = "stderr" if self.verbose < 50 else "stdout"
            s = _name(self.name, None)
            safe_print('Fitting layer %s' % self.name)
            t0 = time()

        if self.dual:

            parallel(delayed(fit_trans)(dir=dir,
                                        case=case,
                                        inst=instance_list,
                                        X=X,
                                        y=y,
                                        idx=tri,
                                        name=self.name)
                     for case, tri, tei, instance_list in self.p)

            parallel(delayed(fit_est)(dir=dir,
                                      case=case,
                                      inst_name=inst_name,
                                      inst=instance,
                                      X=X,
                                      y=y,
                                      pred=P if tei is not None else None,
                                      idx=(tri, tei, self.c[case, inst_name]),
                                      name=self.name,
                                      raise_on_exception=self.raise_,
                                      lim=None, sec=None)
                     for case, tri, tei, instance_list in self.e
                     for inst_name, instance in instance_list)
        else:
            parallel(delayed(_fit)(dir=dir,
                                   case=case,
                                   inst_name=inst_name,
                                   inst=instance,
                                   X=X,
                                   y=y,
                                   pred=P if tei is not None else None,
                                   idx=(tri, tei, self.c[case, inst_name])
                                   if inst_name != '__trans__' else tri,
                                   name=self.layer.name,
                                   raise_on_exception=self.raise_,
                                   lim=self.lim,
                                   sec=self.ival)
                     for case, tri, tei, inst_list in _wrap(self.p) + self.e
                     for inst_name, instance in inst_list)

        # Assemble transformer list
        self.layer.estimators_ = _assemble(dir, self.e, 'e')
        self.layer.preprocessing_ = _assemble(dir, self.p, 't')

        if self.verbose:
            print_time(t0, '%sDone' % s, file=printout)

    def predict(self, X, P, dir, parallel):
        """Predict with fitted layer."""

        self._check_fitted()

        if self.verbose:
            printout = "stderr" if self.verbose < 50 else "stdout"
            s = _name(self.name, None)
            safe_print('Predicting layer %s' % self.name)
            t0 = time()

        # Collect estimators fitted on full data
        prep, ests = self._retrieve('full')

        parallel(delayed(predict_est)(case=case,
                                      tr_list=prep[case],
                                      inst_name=inst_name,
                                      est=est,
                                      xtest=X,
                                      pred=P,
                                      col=col,
                                      name=self.name)
                 for case, (inst_name, est, (_, col)) in ests)

        if self.verbose:
            print_time(t0, '%sDone' % s, file=printout)

    def _check_fitted(self):
        """Utility function for checking that fitted estimators exist."""
        check_is_fitted(self.layer, "estimators_")

        # Check that there is at least one fitted estimator
        if isinstance(self.layer.estimators_, (list, tuple, set)):
            empty = len(self.layer.estimators_) == 0
        elif isinstance(self.layer.estimators_, dict):
            empty = any([len(e) == 0 for e in self.layer.estimators_.values()])
        else:
            # Cannot determine shape of estimators, skip check
            return

        if empty:
            raise NotFittedError("Cannot predict as no estimators were"
                                 "successfully fitted.")

    def _retrieve(self, s):
        """Get transformers and estimators fitted on folds or on full data."""
        cs = self.layer.cases

        if s == 'full':
            # Exploit that instances fitted on full have exact case names
            return (dict([t for t in self.layer.preprocessing_ if t[0] in cs]),
                    [t for t in self.layer.estimators_ if t[0] in cs])

        elif s == 'fold':
            # Exploit that instances fitted on folds have case-fold_num as name
            return (dict([t for t in self.layer.preprocessing_
                          if t[0] not in cs]),
                    [t for t in self.layer.estimators_ if t[0] not in cs])


###############################################################################
def expand_instance_list(instance_list, indexer=None):
    """Build a list of estimation tuples."""
    ls = list()

    if isinstance(instance_list, dict):
        # We need to build fit list on a case basis

        # --- Full data ---
        # Estimators to be fitted on full data. List entries have format:
        # (case, no_train_idx, no_test_idx, est_list)
        # Each est_list have entries (inst_name, cloned_est)
        ls.extend([(case, None, None,
                    [(n, clone(e)) for n, e in instance_list[case]])
                   for case in sorted(instance_list)])

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (case__fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        if indexer is not None:
            fd = [('%s__%i' % (case, i),
                   tri,
                   tei,
                   [('%s__%i' % (n, i), clone(e)) for n, e in
                    instance_list[case]])
                  for case in sorted(instance_list)
                  for i, (tri, tei) in enumerate(indexer.generate())
                  ]
            ls.extend(fd)

    else:
        # No cases to worry about: expand the list of named instance tuples

        # --- Full data ---
        # Estimators to be fitted on full data. List entries have format:
        # (no_case, no_train_idx, no_test_idx, est_list)
        # Each est_list have entries (inst_name, cloned_est)
        ls.extend([(None, None, None,
                    [(n, clone(e)) for n, e in instance_list])])

        # --- Folds ---
        # Estimators to be fitted on each fold. List entries have format:
        # (fold_num, train_idx, test_idx, est_list)
        # Each est_list have entries (inst_name__fol_num, cloned_est)
        if indexer is not None:
            ls.extend([('%i' % i,
                        tri,
                        tei,
                        [('%s__%i' % (n, i), clone(e)) for n, e in
                         instance_list])
                       for i, (tri, tei) in enumerate(indexer.generate())
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
        idx = {(None, inst_name): i for i, (inst_name, _) in
               enumerate(estimators)}

        case_list = [None]
    else:
        # Nested for loop required
        case_list, idx, col = sorted(preprocessing), dict(), 0

        for case in case_list:
            for inst_name, _ in estimators[case]:
                idx[case, inst_name] = col
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
        for inst_name_fold_num, _ in tup[-1]:
            inst_name = inst_name_fold_num.split('__')[0]
            idx[tup[0], inst_name_fold_num] = idx[case, inst_name]

    return idx


def _wrap(folded_list, name='__trans__'):
    """Wrap the folded transformer list.

    wraps a folded transformer list so that the ``tr_list`` appears as
    one estimator with a specified name. Since all ``tr_list``s have the
    same name, it can be used to select a transformation function or an
    estimation function in a combined parallel fitting loop."""
    return [(case, tri, None, [(name, instance_list)]) for
            case, tri, tei, instance_list in folded_list]


def _assemble(dir, instance_list, suffix):
    """Utility for loading fitted instances."""
    if suffix is 't':
        return [(case, pickle_load(os.path.join(dir,
                                                '%s__%s' % (case, suffix))))
                for case, _, _, _ in instance_list]
    else:
        return [(case, pickle_load(os.path.join(dir,
                                   '%s__%s__%s' % (case, inst_name, suffix))))
                for case, _, _, ests in instance_list
                for inst_name, _ in ests]


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


def _slice_array(x, y, idx):
    """Build training array index and slice data."""
    # Have to be careful in prepping data for estimation.
    # We need to slice memmap and convert to a proper array - otherwise
    # transformers can store results memmaped to the cache, which will
    # prevent the garbage collector from releasing the memmaps from memory
    # after estimation

    tri = hstack([arange(t0, t1) for t0, t1 in idx]) \
        if idx is not None else None

    x = x[tri] if tri is not None else x
    y = asarray(y[tri]) if idx is not None else asarray(y)

    if x.__class__.__name__[:3] not in ['csr', 'csc', 'coo', 'dok']:
        # numpy asarray does not work with scipy sparse. Current experimental
        # solution is to just leave them as is.
        x = asarray(x)

    return x, y


###############################################################################
def predict_est(case, tr_list, inst_name, est, xtest, pred, col, name):
    """Method for predicting with fitted transformers and estimators."""
    # Transform input
    for tr_name, tr in tr_list:
        xtest = _transform_tr(xtest, tr, tr_name, case, name)

    # Predict into memmap
    # Here, we coerce errors on failed predictions - all predictors that
    # survive into the estimators_ attribute of a layer should be able to
    # predict, otherwise the subsequent layer will get corrupt input.
    pred[:, col] = _predict_est(xtest, est, True, inst_name, case, name)


def fit_trans(dir, case, inst, X, y, idx, name):
    """Fit transformers and write to cache."""
    x, y = _slice_array(X, y, idx)

    out = []
    for tr_name, tr in inst:
        # Fit transformer
        tr = _fit_tr(x, y, tr, tr_name, case, name)

        # If more than one step, transform input for next step
        if len(inst) > 1:
            x = _transform_tr(x, tr, tr_name, case, name)
        out.append((tr_name, tr))

    # Write transformer list to cache
    f = os.path.join(dir, '%s__t' % case)
    pickle_save(out, f)


def fit_est(dir, case, inst_name, inst, X, y, pred, idx, raise_on_exception,
            name, lim, sec):
    """Fit estimator and write to cache along with predictions."""
    # Have to be careful in prepping data for estimation.
    # We need to slice memmap and convert to a proper array - otherwise
    # estimators can store results memmaped to the cache, which will
    # prevent the garbage collector from releasing the memmaps from memory
    # after estimation
    x, y = _slice_array(X, y, idx[0])

    # Load transformers
    f = os.path.join(dir, '%s__t' % case)
    tr_list = _load_trans(f, case, lim, sec, raise_on_exception)

    # Transform input
    for tr_name, tr in tr_list:
        x = _transform_tr(x, tr, tr_name, case, name)

    # Fit estimator
    est = _fit_est(x, y, inst, raise_on_exception, inst_name, case, name)

    # Predict if asked
    # The predict loop is kept separate to allow overwrite of x, thus keeping
    # only one subset of X in memory at any given time
    if idx[1] is not None:
        tei = idx[1]
        col = idx[2]

        x = X[tei[0]:tei[1]]
        if x.__class__.__name__[:3] not in ['csr', 'csc', 'coo', 'dok']:
            x = asarray(x)

        for tr_name, tr in tr_list:
            x = _transform_tr(x, tr, tr_name, case, name)

        pred[tei[0]:tei[1], col] = \
            _predict_est(x, est, raise_on_exception, inst_name, case, name)

    # We drop tri from index and only keep tei if any predictions were made
        idx = idx[1:]
    else:
        idx = (None, idx[2])

    f = os.path.join(dir, '%s__%s__e' % (case, inst_name))
    pickle_save((inst_name, est, idx), f)


def _fit(**kwargs):
    """Wrapper to select fit_est or fit_trans."""
    f = fit_trans if kwargs['inst_name'] == '__trans__' else fit_est
    f(**{k: v for k, v in kwargs.items() if k in f.__code__.co_varnames})


###############################################################################
def _load_trans(f, case, lim, s, raise_on_exception):
    """Try loading transformers, and handle exception if not ready yet."""
    try:
        # Assume file exists
        return pickle_load(f)
    except FileNotFoundError or TypeError as exc:
        msg = str(exc)
        error_msg = ("The file %s cannot be found after %i seconds of "
                     "waiting. Check that time to fit transformers is "
                     "sufficiently fast to complete fitting before "
                     "fitting estimators. Consider reducing the "
                     "preprocessing intensity in the ensemble, or "
                     "increase the '__lim__' attribute to wait extend "
                     "period of waiting on transformation to complete."
                     " Details:\n%r")

        if raise_on_exception:
            # Raise error immediately
            raise ParallelProcessingError(error_msg % msg)

        # Else, check intermittently until limit is reached
        ts = time()
        while not os.path.exists(f):
            sleep(s)
            if time() - ts > lim:
                if raise_on_exception:
                    raise ParallelProcessingError(error_msg % msg)

                warnings.warn("Transformer %s not found in cache (%s). "
                              "Will check every %.1f seconds for %i seconds "
                              "before aborting. " % (case, f, s, lim),
                              ParallelProcessingWarning)

                raise_on_exception = True
                ts = time()

        return pickle_load(f)


def _fit_tr(x, y, tr, tr_name, case, layer_name):
    """Wrapper around try-except block for fitting transformer."""
    try:
        return tr.fit(x, y)
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


def _fit_est(x, y, est, raise_on_exception, inst_name, case, layer_name):
    """Wrapper around try-except block for estimator fitting."""
    try:
        return est.fit(x, y)
    except Exception as e:
        s = _name(layer_name, case)

        if raise_on_exception:
            raise FitFailedError("%sCould not fit estimator '%s'. "
                                 "Details:\n%r" % (s, inst_name, e))

        msg = "%sCould not fit estimator '%s'. Will drop from " \
              "ensemble. Details:\n%r"
        warnings.warn(msg % (s, inst_name, e), FitFailedWarning)


def _predict_est(x, est, raise_on_exception, inst_name, case, layer_name):
    """Wrapper around try-except block for estimator predictions."""
    try:
        return est.predict(x)
    except Exception as e:
        s = _name(layer_name, case)

        if raise_on_exception:
            raise PredictFailedError("%sCould not predict with estimator '%s'."
                                     " Details:\n%r" % (s, inst_name, e))

        msg = "%sCould not predict with estimator '%s'. Predictions will be" \
              "0. Details:\n%r"
        warnings.warn(msg % (s, inst_name, e), PredictFailedWarning)
