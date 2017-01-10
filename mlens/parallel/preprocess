#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:01:02 2017

author: Sebastian Flennerhag
date: 10/01/2017
Functions for parallelized preprocessing
"""

from ..utils import _slice
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import KFold


def _preprocess_pipe(xtrain, ytrain, xtest, steps, fit, p_name=None,
                     dry_run=False, return_estimator=False):
    """ Function to fit and transform all data with preprocessing pipeline """

    for step in steps:
        if fit:
            step.fit(xtrain, ytrain)
        xtrain = step.transform(xtrain)
        if xtest is not None:
            xtest = step.transform(xtest)

    if not dry_run:
        if return_estimator:
            out = [steps, xtrain]
        else:
            out = [xtrain]
        if xtest is not None:
            out.append(xtest)
        if p_name is not None:
            out.append(p_name)

        return out


def _preprocess_fold(X, y, train_idx, test_idx, preprocess_case=[],
                     p_name=None, fit=True):
    """ Function to fit and transform a fold with a preprocessing pipeline """

    xtrain, xtest = _slice(X, train_idx), _slice(X, test_idx)

    try:
        ytrain, ytest = _slice(y, train_idx), _slice(y, test_idx)
    except:
        ytrain, ytest = None, None

    if len(preprocess_case) != 0:
        out = _preprocess_pipe(xtrain, ytrain, xtest, preprocess_case, fit)
    else:
        out = [xtrain, xtest]

    out += [ytrain, ytest, test_idx, p_name]

    return out


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
                     shuffle=False, n_jobs=-1, verbose=False):
    """ Pre-make preprecessing cases over cv folds (incl no preprocessing)"""

    kfold = KFold(folds, shuffle=shuffle)

    # Check if there is at least one preprocessing pipeline
    if len(list(preprocessing)) != 0:
        dout = Parallel(n_jobs=n_jobs, verbose=verbose)(
                        delayed(_preprocess_fold)(X, y, train_idx, test_idx,
                                                  process_case, pname, fit=fit)
                        for train_idx, test_idx in kfold.split(X)
                        for pname, process_case in preprocessing)
    else:
        dout = Parallel(n_jobs=n_jobs, verbose=verbose)(
                        delayed(_preprocess_fold)(X, y, train_idx, test_idx,
                                                  fit=fit)
                        for train_idx, test_idx in kfold.split(X))
    return dout
