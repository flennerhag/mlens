#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

from sklearn.base import clone
import numpy as np
from pandas import DataFrame
from ..ensemble._clone import _clone_preprocess_cases
from ..parallel import preprocess_folds, cross_validate
from sklearn.pipeline import Pipeline
from time import time
import sys


class Evaluator(object):
    '''
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
    preprocessing: dict, default={}
        dictionary of lists with preprocessing pipelines to fit models on.
        Each pipeline will be used to generate k folds that are stored, hence
        with large data running several cases with many cv folds can require
        considerable memory. preprocess should be of the form:
            P = {'case-1': [step1, step2], ...}
    cv : int, default=2
        cross validation folds to use. Currently standard kfold implemented
    shuffle : bool, default=True,
        whether to shuffle data before creating folds
    scoring : func
        scoring function that follows sklearn API,
        i.e. score = scoring(estimator, X, y)
    summary_df : bool, default=True
        whether to return summary data in a pandas DataFrame, else as a dict
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
    summary_ : dict, DataFrame
        Summary output that shows best scores, times, params, estimators
    summary_dict_ : dict, default=None,
        if summary_df=True, the full summary dict is preserved in summary_dict_
    cv_results_ : dict
        dictionary containing all data for every fold, every param draw, and
        every estimator

    Methods
    --------
    preprocess :
        Preprocess data according to specified pipeliens and cv folds.
        Preprocessed data is stored in class instance to allow for repeated
        evaluation of estimators
    evaluate : estimators, param_dicts, n_iter, reset_preprocess
        Method to run grid search on a set of estimators with given param_dicts
        for n_iter iterations. Set reset_preprocess to True to regenerate
        preprocessed data
    '''

    def __init__(self, X, y, preprocessing, scoring, cv=10, shuffle=True,
                 summary_df=True, random_state=True, n_jobs_preprocessing=-1,
                 n_jobs_estimators=-1, verbose=0):
        self.X = X.copy()
        self.y = y.copy()
        self.cv = cv
        self.shuffle = shuffle
        self.summary_df = summary_df
        self.n_jobs_preprocessing = n_jobs_preprocessing
        self.n_jobs_estimators = n_jobs_estimators
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.preprocessing = preprocessing
        self._printout = sys.stdout if self.verbose > 50 else sys.stderr

    def preprocess(self):
        ''' Method for preprocessing data separately from estimator
            evaluation. Helpful if preprocessing is costly relative to
            estimator fitting and flexibility is needed in evaluating
            estimators. Examples include fitting base estimators as part of
            preprocessing, to evaluate suitabe meta estimators in ensembles.'''

        self.preprocessing_ = _clone_preprocess_cases(self.preprocessing)

        self.dout = preprocess_folds(self.preprocessing_, self.X, self.y,
                                     self.cv, fit=True, return_idx=False,
                                     shuffle=self.shuffle,
                                     random_state=self.random_state,
                                     n_jobs=self.n_jobs_preprocessing,
                                     verbose=self.verbose)
        return self

    def evaluate(self, estimators, param_dicts, n_iter=2,
                 reset_preprocess=False):
        '''
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
        Returns
        ---------
        '''

        self.n_iter = n_iter
        self.estimators_ = estimators
        self.param_dicts_ = param_dicts

        # ===== Preprocess if necessary or requested =====
        if not hasattr(self, 'dout') or reset_preprocess:
            self.preprocess()

        # ===== Metric data to be stored for each fit =====
        self._metrics = ['test_score', 'train_score', 'time']

        # ===== Generate n_iter param dictionaries for each estimator =====
        self.param_sets_ = self._param_sets()

        # ===== Set up cv results dictionary =====
        self.cv_results_ = self._set_up_cv()

        # ===== Cross Validate =====
        if self.verbose > 0:
            ttot = self._print_start(estimators)

        out = cross_validate(self.estimators_, self.param_sets_, self.dout,
                             self.scoring, self.n_jobs_estimators,
                             self.verbose)

        # ===== Create summary statistics =====
        self._gen_summary(out)

        # ===== Job complete =====
        if self.verbose > 0:
            res, secs = divmod(time() - ttot, 60)
            hours, mins = divmod(res, 60)
            print('Evaluation done | %02d:%02d:%02d\n' % (hours, mins, secs),
                  file=self._printout)
        return self

    def _gen_summary(self, out):
        self._store_cv(out)
        stats = self._fold_stats()
        self.summary_ = self._summarize(stats)
        if self.summary_df:
            self.summary_dict_ = self.summary_
            df = DataFrame(self.summary_).T.sort_values(by='best_test_score_mean',
                                                        ascending=False)
            self.summary_ = df.loc[:, ['best_test_score_mean',
                                       'best_test_score_std',
                                       'train_score_mean', 'train_score_std',
                                       'score_time', 'best_params',
                                       'best_estimator', 'best_draw_idx']]

    def _store_cv(self, out):
        for tup in out:
            (est_name, test_score, train_score, time, draw) = tup

            for key, val in zip(['test_score', 'train_score', 'time'],
                                [test_score, train_score, time]):
                self.cv_results_[est_name][draw][key].append(val)

    def _draw_params(self, est_name):
        params = {}
        for param, dist in self.param_dicts_[est_name].items():
            params[param] = dist.rvs(1, random_state=self.random_state)[0]
        return params

    def _param_sets(self):
        param_set = {}
        for est_name, _ in self.estimators_.items():
            param_set[est_name] = []
            for _ in range(self.n_iter):
                param_set[est_name].append(self._draw_params(est_name))
        return param_set

    def _set_up_cv(self):
        C = {}
        if len(self.preprocessing_) is 0:
            for est_name, est in self.estimators_.items():
                    name = est_name
                    C[name] = {}
                    for i, params in enumerate(self.param_sets_[est_name]):

                        # Generate full models and store
                        e = clone(est)
                        e.set_params(**params)

                        C[name][i+1] = {'params': params, 'estimator': e}
                        for metric in self._metrics:
                            C[name][i+1][metric] = []
        else:
            for est_name, est in self.estimators_.items():
                for p_name, process_case in self.preprocessing_:
                    name = est_name + '_' + p_name
                    C[name] = {}
                    for i, params in enumerate(self.param_sets_[est_name]):

                        # Generate full models and store
                        e = clone(est)
                        e.set_params(**params)

                        prep = [('prep_' + str(i+1), step)
                                for i, step in enumerate(process_case)]

                        if isinstance(e, Pipeline):
                            # add preprocessing steps to pipeline
                            e = Pipeline(prep + e.steps)
                        else:
                            # create new pipeline
                            e = Pipeline(prep + [('est', e)])

                            params_original = params
                            params = {}
                            for key, val in params_original.items():
                                params['est__' + key] = val

                        C[name][i+1] = {'params': params, 'estimator': e}
                        for metric in self._metrics:
                            C[name][i+1][metric] = []
        return C

    def _fold_stats(self):
        statistics = {}
        for est_name in self.cv_results_.keys():
            statistics[est_name] = {'mean_test_score': [],
                                    'mean_test_score_std': [],
                                    'mean_train_score': [],
                                    'mean_train_score_std': [],
                                    'mean_time': [], 'params': [],
                                    'estimator': []}
            for draw, results in self.cv_results_[est_name].items():

                t = np.mean(results['time'])
                testm = np.mean(results['test_score'])
                tests = np.std(results['test_score'])
                trainm = np.mean(results['train_score'])
                trains = np.std(results['train_score'])
                p = self.cv_results_[est_name][draw]['params']
                e = self.cv_results_[est_name][draw]['estimator']

                statistics[est_name]['params'].append(p)
                statistics[est_name]['estimator'].append(e)
                for val, nm in zip([t, testm, tests, trainm, trains],
                                   ['mean_time', 'mean_test_score',
                                    'mean_test_score_std', 'mean_train_score',
                                    'mean_train_score_std']):
                    self.cv_results_[est_name][draw][nm] = val
                    statistics[est_name][nm].append(val)
        return statistics

    def _summarize(self, stats):
        S = {}
        for model, model_stats in stats.items():
            S[model] = {}

            best_idx = np.argmax(model_stats['mean_test_score'])
            S[model]['best_test_score_mean'] = \
                model_stats['mean_test_score'][best_idx]
            S[model]['best_test_score_std'] = \
                model_stats['mean_test_score_std'][best_idx]
            S[model]['train_score_mean'] = \
                model_stats['mean_train_score'][best_idx]
            S[model]['train_score_std'] = \
                model_stats['mean_train_score_std'][best_idx]
            S[model]['score_time'] = model_stats['mean_time'][best_idx]
            S[model]['best_params'] = model_stats['params'][best_idx]
            S[model]['best_estimator'] = model_stats['estimator'][best_idx]
            S[model]['best_draw_idx'] = best_idx + 1
        return S

    def _print_start(self, estimators):
        ttot = time()
        msg = ('Evaluating %i models, with %i parameter draws and %i' +
               ' preprocessing options, over %i CV folds, totalling fits %i.')
        e = len(estimators)
        try:
            p = max(len(self.preprocessing_), 1)
        except Exception:
            p = 0
        tot = e * max(1, p) * self.n_iter * self.cv
        print(msg % (e, self.n_iter, p, self.cv, tot), file=self._printout)
        return ttot
