#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:01:02 2017

author: Sebastian Flennerhag
date: 10/01/2017
Functions for parallelized preprocessing
"""

from ..utils import _slice


def _preprocess_pipe(xtrain, ytrain, xtest, steps, fit, p_name=None,
                     dry_run=False, return_estimator=False):
    """ Function to fit and transform all data with preprocessing pipeline """

    for step in steps:
        if fit:
            step.fit(xtrain, ytrain)
        xtrain = step.transform(xtrain)
        if xtest is not None:
            xtest = step.transform(xtest)

    if dry_run:
        return
    else:
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
                     p_name=None, fit=True, return_idx=True):
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

    out += [ytrain, ytest]
    if return_idx:
        out.append(test_idx)
    out.append(p_name)

    return out
