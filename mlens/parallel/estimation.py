"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base class for estimation.
"""

import numpy as np
from abc import ABCMeta, abstractmethod

from ._base_functions import fit, predict, transform, construct_args
from ..utils import check_is_fitted, print_time, safe_print
from ..utils.exceptions import NotFittedError, ParallelProcessingWarning

try:
    from time import perf_counter as time_
except ImportError:
    from time import time as time_

import warnings


FUNCS = {'fit': fit,
         'predict': predict,
         'predict_proba': predict,
         'transform': transform
         }


class BaseEstimator(object):

    """Base class for estimating a layer in parallel.

    Estimation class to be used as based for a layer estimation engined that
    is callable by the :class:`ParallelProcess` job manager.

    A subclass must implement a constructor that accepts the following args:
        - ``job`` : the :class:`Job` instance containing relevant data
        - ``layer``: the :class:`Layer` instance to estimate
        - ``n``: the position in the :class:`LayerContainer` stack of the layer

    as well as a ``run`` method that accepts a :class:`Parallel` instance.

    Parameters
    ----------
    layer : :class:`Layer`
        layer to be estimated.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, layer):
        self.layer = layer

    def __call__(self, parallel):
        """Defines the job to complete.

        Parameters
        ----------
        parallel : object
            :class:`Parallel` instance.
        """
        if self.layer.verbose:
            printout = "stderr" if self.layer.verbose < 50 else "stdout"
            safe_print('Processing %s' % self.layer.name, file=printout)
            t0 = time_()

        self.run(parallel)

        if self.layer.verbose:
            print_time(t0, '%s Done' % self.layer.name, file=printout)

    @abstractmethod
    def run(self, parallel):
        """Method for executing estimation.

        Default method relies on the default constructor. Both can be replaced
        if desired.

        Parameters
        ----------
        parallel : object
            :class:`Parallel` instance.
        """
        self.execute(self, parallel=parallel, **self.args)

    def _default_initialization(self, job, n):
        """Utility method for default initialization scheme."""
        self.dir = job.dir
        self.execute = FUNCS[job.j]
        self.args = construct_args(self.execute, job, n)

    def _build_scores(self, s):
        """Build a cv-score mapping."""
        scores = dict()

        # Build shell dictionary with main estimators as keys
        for k, v in s[:self.layer.n_pred]:
            case_name, est_name = k.split('___')

            if case_name == '':
                name = est_name
            else:
                name = '%s__%s' % (case_name, est_name)

            scores[name] = []

        # Populate with list of scores from folds
        for k, v in s[self.layer.n_pred:]:
            case_name, est_name = k.split('___')

            est_name = '__'.join(est_name.split('__')[:-1])

            if '__' not in case_name:
                name = est_name
            else:
                case_name = case_name.split('__')[0]
                name = '%s__%s' % (case_name, est_name)

            scores[name].append(v)

        # Aggregate to get cross-validated mean scores
        for k, v in scores.items():
            if None in v or len(v) == 0:
                continue

            try:
                scores[k] = (np.mean(v), np.std(v))
            except Exception as exc:
                warnings.warn("Aggregating scores failed. Scores:\n%r\n"
                              "Details: %r" % (v, exc),
                              ParallelProcessingWarning)
        return scores

    def _check_fitted(self):
        """Utility function for checking that fitted estimators exist."""
        check_is_fitted(self.layer, "estimators_")

        assert isinstance(self.layer.estimators_, list)
        if len(self.layer.estimators_) == 0:
            raise NotFittedError("No estimators successfully fitted.")

    def _retrieve(self, s):
        """Get transformers and estimators fitted on folds or on full data."""
        n_pred = self.layer.n_pred
        n_prep = max(self.layer.n_prep, 1)

        if s == 'full':
            # If full, grab the first n_pred estimators, and the first
            # n_prep preprocessing pipelines, which are fitted on
            # the full training data. We take max on n_prep to avoid getting
            # empty preprocessing_ slice when n_prep = 0 when no preprocessing.
            ests = self.layer.estimators_[:n_pred]

            if self.layer.preprocessing_ is None:
                prep = None
            else:
                prep = dict(self.layer.preprocessing_[:n_prep])

        elif s == 'fold':
            # If fold, grab the estimators after n_pred, and the preprocessing
            # pipelines after n_prep, which are fitted on folds of the
            # training data.
            ests = self.layer.estimators_[n_pred:]

            if self.layer.preprocessing_ is None:
                prep = None
            else:
                prep = dict(self.layer.preprocessing_[n_prep:])

        else:
            raise ValueError("Argument not understood. Only 'full' and 'fold' "
                             "are acceptable argument values.")

        return prep, ests
