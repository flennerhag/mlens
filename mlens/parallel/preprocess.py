#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
Functions for parallelized preprocessing
"""

from __future__ import division, print_function

from ._preprocess_functions import _preprocess_pipe, _preprocess_fold
from joblib import Parallel, delayed
from sklearn.model_selection import KFold


def preprocess_pipes(preprocessing, X, y=None, fit=True, dry_run=False,
                     return_estimators=False, n_jobs=-1, verbose=False):
    """Pre-make preprocessing cases for all data (no folds)"""
    dout = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(_preprocess_pipe)(X, y, None, process_case,
                                              fit, p_name, dry_run,
                                              return_estimators)
                    for p_name, process_case in preprocessing)
    return dout


def preprocess_folds(preprocessing, X, y=None, folds=None, fit=True,
                     shuffle=False, random_state=None, return_idx=True,
                     n_jobs=-1, verbose=False):
    """Pre-make preprecessing cases over cv folds (incl no preprocessing)"""
    if isinstance(folds, int):
        kfold = KFold(folds, shuffle=shuffle, random_state=random_state)
    else:
        kfold = folds

    # Safety check to ensure for loop generate folds when no preprocessing
    # was desired
    if (preprocessing is None) or (len(preprocessing) == 0):
        preprocessing = [None]

    dout = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(_preprocess_fold)(X, y, indices,
                                              process_case, fit=fit,
                                              return_idx=return_idx)
                    for indices in kfold.split(X)
                    for process_case in preprocessing)
    return dout
