#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
data: 10/01/2017
"""

from time import time
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
import warnings


def fit_score(est, est_name, params, scoring, xtrain, xtest, ytrain, ytest,
              p_name=None, draw=None, ret_time=False, ret_train=False):
    """ Score an estimator with given parameters on train and test set """
    try:
        est = clone(est)
        est.set_params(**params)
        t0 = time()
        est.fit(xtrain, ytrain)
        t = time() - t0
    except KeyError as exc:
        msg = "Could not fit estimator [%s]. Details: \n%r"
        warnings.warn(msg % (est_name, exc), FitFailedWarning)

    test_sc = scoring(est, xtest, ytest)
    train_sc = scoring(est, xtrain, ytrain)

    if p_name is not None:
        est_name += '_' + p_name

    out = [est_name, test_sc]

    if ret_train:
        out.append(train_sc)
    if ret_time:
        out.append(t)
    if draw is not None:
        out.append(draw + 1)
    return out
