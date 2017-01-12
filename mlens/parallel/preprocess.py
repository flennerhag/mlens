#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:01:02 2017

author: Sebastian Flennerhag
date: 10/01/2017
Functions for parallelized preprocessing
"""

from ._preprocess_functions import _preprocess_pipe, _preprocess_fold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import KFold


def preprocess_pipes(preprocessing, X, y=None, fit=True, dry_run=False,
                     return_estimators=False, n_jobs=-1, verbose=False):
    """ Pre-make preprocessing cases for all data (no folds)"""

    dout = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(_preprocess_pipe)(X, y, None, process_case,
                                              fit, p_name, dry_run,
                                              return_estimators)
                    for p_name, process_case in preprocessing)
    return dout


def preprocess_folds(preprocessing, X, y=None, folds=None, fit=True,
                     shuffle=False, random_state=None, return_idx=True,
                     n_jobs=-1, verbose=False):
    """ Pre-make preprecessing cases over cv folds (incl no preprocessing)"""

    kfold = KFold(folds, shuffle=shuffle, random_state=random_state)

    # Check if there is at least one preprocessing pipeline
    if len(list(preprocessing)) != 0:
        dout = Parallel(n_jobs=n_jobs, verbose=verbose)(
                        delayed(_preprocess_fold)(X, y, train_idx, test_idx,
                                                  process_case, pname, fit=fit,
                                                  return_idx=return_idx)
                        for train_idx, test_idx in kfold.split(X)
                        for pname, process_case in preprocessing)
    else:
        dout = Parallel(n_jobs=n_jobs, verbose=verbose)(
                        delayed(_preprocess_fold)(X, y, train_idx, test_idx,
                                                  fit=fit)
                        for train_idx, test_idx in kfold.split(X))
    return dout
