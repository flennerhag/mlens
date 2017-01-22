#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
Class for paralellized tuning a set of estimators that share a common
preprocessing pipeline that must be fitted on each training fold. This
implementation improves on standard grid search by avoiding fitting the
preprocessing pipeline for every estimators, and allowing several alternative
preprocessing cases to be evaluated. Tuning information for all estimators
and all cases are accessibly stored in a summary attribute.
"""

from __future__ import division, print_function

import numpy as np
from pandas import DataFrame, Series
from ..ensemble._clone import _clone_preprocess_cases
from ..parallel import preprocess_folds, cross_validate
from time import time
import sys


class Evaluator(object):

    """Class for evaluating a set of estimators and preprocessing pipelines

    Evaluator class that allows user to evaluate several models simoultanously
    across a set of pre-specified pipelines. The class is useful for comparing
    a set of estimators when several preprocessing pipelines have potential.
    By fitting all estimators on the same folds, number of fit can be greatly
    reduced as compared to pipelining each estimator and gitting them in an
    sklearn grid search. If preprocessing is time consuming, the evaluator
    class can be order of magnitued faster than a standard gridsearch.

    If the user in unsure about what estimators to fit, the preprocess method
    can be used to preprocess data, after which the evuate method can be run
    any number of times upon the pre-made folds for various configurations of
    parameters. Current implementation only accepts randomized grid search.

    Parameters
    -----------
    X : array-like, shape=[n_samples, n_features]
        input data to train estimators on.
    y : array-like, shape=[n_samples,]
        output data to train estimators on.
    scoring : func
        scoring function that follows sklearn API,
        i.e. score = scoring(estimator, X, y)
    error_score : int,
        score to assign when estimator fit fails
    preprocessing: dict, default=None
        dictionary of lists with preprocessing pipelines to fit models on.
        Each pipeline will be used to generate k folds that are stored, hence
        with large data running several cases with many cv folds can require
        considerable memory. preprocess should be of the form:
            P = {'case-1': [step1, step2], ...}
    cv : int, default=2
        cross validation folds to use. Either pass a KFold class object that
        accepts as ``split`` method, or the number of folds in standard KFold
    shuffle : bool, default=True,
        whether to shuffle data before creating folds
    random_state : int, default=None
        seed for creating folds
    n_jobs_preprocessing : int, default=-1
        number of CPU cores to use for preprocessing of folds
    n_jobs_estimators : int, default=-1
        number of CPU cores to use for grid search (estimator fitting)
    verbose : bool, int, default=False
        level of printed output.

    Attributes
    -----------
    summary_ : DataFrame
        Summary output that shows data for best mean test scores, such as
        test and train scores, std, fit times, and params
    cv_results_ : DataFrame
        a table of data from each fit. Includes mean and std of test and train
        scores and fit times, as well as param draw index and parameters.
    best_index : ndarray,
        an array of index keys for best estimator in ``cv_results_``

    Methods
    --------
    preprocess : None
        Preprocess data according to specified pipelines and cv folds.
        Preprocessed data is stored in class instance to allow for repeated
        evaluation of estimators
    evaluate : estimators, param_dicts, n_iter, reset_preprocess
        Method to run grid search on a set of estimators with given param_dicts
        for n_iter iterations. Set reset_preprocess to True to regenerate
        preprocessed data
    """

    def __init__(self, scoring, preprocessing=None, cv=10, shuffle=True,
                 random_state=None, n_jobs_preprocessing=-1,
                 error_score=-99, n_jobs_estimators=-1, verbose=0):
        self.cv = cv
        self.shuffle = shuffle
        self.n_jobs_preprocessing = n_jobs_preprocessing
        self.n_jobs_estimators = n_jobs_estimators
        self.error_score = error_score
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.preprocessing = preprocessing

    def preprocess(self, X, y):
        """Preprocess folds

        Method for preprocessing data separately from estimator
        evaluation. Helpful if preprocessing is costly relative to
        estimator fitting and flexibility is needed in evaluating
        estimators. Examples include fitting base estimators as part of
        preprocessing, to evaluate suitabe meta estimators in ensembles.
        """
        self.preprocessing_ = _clone_preprocess_cases(self.preprocessing)

        if self.verbose > 0:
            printout = sys.stdout if self.verbose > 50 else sys.stderr
            ttot = self._print_prep_start(self.preprocessing_, printout)

        self.dout = preprocess_folds(self.preprocessing_, X.copy(), y.copy(),
                                     self.cv, fit=True, return_idx=False,
                                     shuffle=self.shuffle,
                                     random_state=self.random_state,
                                     n_jobs=self.n_jobs_preprocessing,
                                     verbose=self.verbose)

        if self.verbose > 0:
            res, sec = divmod(time() - ttot, 60)
            hrs, mins = divmod(res, 60)
            print('Preprocessing done | %02d:%02d:%02d\n' % (hrs, mins, sec),
                  file=printout)
            printout.flush()

        return self

    def evaluate(self, X, y, estimators, param_dicts, n_iter=2,
                 reset_preprocess=False, flush_preprocess=False):
        """Evaluate estimators

        Function for evaluating a list of functions, potentially with various
        preprocessing pipelines. This method improves fit time of regular grid
        search of a list of estimators since preprocessing is done once
        for each fold, rather than for each fold and estimator.
        [Note: if preprocessing was performed previous to calling evaluate,
         preprocessed folds will be used. To re-run preprocessing, set
         reset_preprocess to True.]

        Parameters
        ----------
        estimators, dict
            set of estimators to use: estimators={'est1': est(), ...}
        param_dicts, dict
            param_dicts for estimators. Current implementation only supports
            randomized grid search, where passed distributions accept the
            .rvs() method. See sklearn.model_selection.RandomizedSearchCV for
            details.Form: param_dicts={'est1': {'param1': dist}, ...}
        n_ier : int
            number of parameter draws
        reset_preprocess : bool, default=False
            set to True to regenerate preprocessed folds
        flush_preprocess : bool, default=False
            set to True to drop preprocessed data. Useful if memory requirement
            is large.
        Returns
        ---------
        self : obj
            class instance with stored evaluation data
        """
        self.n_iter = n_iter
        self.estimators_ = estimators
        self.param_dicts_ = param_dicts

        # ===== Preprocess if necessary or requested =====
        if not hasattr(self, 'dout') or reset_preprocess:
            self.preprocess(X, y)

        # ===== Generate n_iter param dictionaries for each estimator =====
        self.param_sets_, param_map = self._param_sets()

        # ===== Cross Validate =====
        if self.verbose > 0:
            printout = sys.stdout if self.verbose > 50 else sys.stderr
            ttot = self._print_eval_start(estimators, self.preprocessing_,
                                          printout)

        out = cross_validate(self.estimators_, self.param_sets_, self.dout,
                             self.scoring, self.error_score,
                             self.n_jobs_estimators, self.verbose)

        # ===== Create summary statistics =====
        self.cv_results_, self.summary_, self.best_idx_ = \
            self._results(out, param_map)

        # ===== Job complete =====
        if flush_preprocess:
            del self.dout

        if self.verbose > 0:
            res, secs = divmod(time() - ttot, 60)
            hours, mins = divmod(res, 60)
            print('Evaluation done | %02d:%02d:%02d\n' % (hours, mins, secs),
                  file=printout)
            printout.flush()

        return self

    # Auxilliary function for param draws and results mapping
    def _draw_params(self, est_name):
        """Draw a list of param dictionaries for estimator"""
        # Set up empty list of parameter setting
        param_draws = [{} for _ in range(self.n_iter)]

        # Fill list of parameter settings by param
        for param, dist in self.param_dicts_[est_name].items():

            draws = dist.rvs(self.n_iter, random_state=self.random_state)

            for i, draw in enumerate(draws):
                param_draws[i][param] = draw

        return param_draws

    def _param_sets(self):
        """For each estimator, create a mapping of parameter draws"""
        param_sets = {}  # dict with list of param settings for each est
        param_map = {}   # dict with param settings for each est_prep pair

        # Create list of param settings for each estimator
        for est_name, _ in self.estimators_.items():
            param_sets[est_name] = self._draw_params(est_name)

        # Flatten list to param draw mapping for each preprocessing case
        for est_name, param_draws in param_sets.items():
            for draw, params in enumerate(param_draws):
                for case in self.preprocessing.keys():
                    param_map[(est_name + '-' + case, draw + 1)] = params

        return param_sets, param_map

    def _results(self, out, param_map):
        # Construct a results dataframe for each param draw
        out = DataFrame(out, columns=['estimator', 'test_score',
                                      'train_score', 'time',
                                      'param_draw', 'params'])

        # Get mean scores for each param draw
        cv_results = out.groupby(['estimator', 'param_draw']).agg(['mean',
                                                                   'std'])

        # Append param settings
        param_map = Series(param_map)
        param_map.index.names = ['estimator', 'param_draw']
        cv_results['params'] = param_map.loc[cv_results.index]

        # Create summary table of best scores
        ts_id = ('test_score', 'mean')
        best_score = cv_results.loc[:, ts_id].groupby(level=0).apply(np.argmax)
        best_idx = best_score.values
        summary = cv_results.loc[best_idx].reset_index(1, drop=True)

        return cv_results, summary, best_idx

    def _print_prep_start(self, preprocessing, printout):
        ttot = time()
        msg = 'Preprocessing %i preprocessing pipelines over %i CV folds'

        try:
            p = max(len(preprocessing), 1)
        except Exception:
            p = 0

        c = self.cv if isinstance(self.cv, int) else self.cv.n_splits

        print(msg % (p, c), file=printout)
        printout.flush()
        return ttot

    def _print_eval_start(self, estimators, preprocessing, printout):
        ttot = time()
        msg = ('Evaluating %i models for %i parameter draws over %i' +
               ' preprocessing pipelines and %i CV folds, totalling %i fits')
        e = len(estimators)
        try:
            p = max(len(preprocessing), 1)
        except Exception:
            p = 0

        c = self.cv if isinstance(self.cv, int) else self.cv.n_splits

        tot = e * max(1, p) * self.n_iter * c
        print(msg % (e, self.n_iter, p, c, tot), file=printout)
        printout.flush()
        return ttot
