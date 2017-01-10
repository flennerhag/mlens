#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
data: 10/01/2017
Support functions for cloning ensemble estimators
"""

from ._setup import check_estimators
from sklearn.base import clone


def _clone_base_estimators(base_estimators, as_dict=True):
    """ Created named clones of base estimators for fitting """
    if as_dict:
        return {case: [(est_name, clone(est)) for est_name, est in
                       estimators]
                for case, estimators in base_estimators}
    else:
        return [(case, [clone(est) for est in
                        check_estimators(estimators)])
                for case, estimators in base_estimators]


def _clone_preprocess_cases(preprocess):
    """ Created named clones of base preprocessing pipes for fitting """
    return [(case, [clone(trans) for trans in
            check_estimators(process_pipe)])
            for case, process_pipe in preprocess]
