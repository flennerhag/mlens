#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
Support functions for naming estimators to enable ensemble parameter mapping
"""

from __future__ import division, print_function

from sklearn.pipeline import Pipeline, _name_estimators


def name_estimators(estimators, prefix='', suffix=''):
    """ Function for creating dict with named estimators for get_params """
    if len(estimators) == 0:
        return {}
    else:
        if isinstance(estimators[0], tuple):
            named_estimators = {prefix + est_name + suffix: est for
                                est_name, est in estimators}
        else:
            named_estimators = {prefix + est_name + suffix: est for
                                est_name, est in _name_estimators(estimators)}
        return named_estimators


def name_base(base_pipelines):
    """ Function for naming estimators and transformers for base pipelines """
    # if a list is passed, assume it is a list of base estimators
    # i.e. no preprocessing
    if isinstance(base_pipelines, list):
        return name_estimators(base_pipelines)
    else:
        named_pipelines = {}
        for p_name, pipe in base_pipelines.items():

            if isinstance(pipe, Pipeline):
                pipe = pipe.steps

            for phase in pipe:
                name = p_name + '-' if len(p_name) > 0 else ''
                named_pipelines.update(name_estimators(phase, name))

    return named_pipelines


def check_estimators(estimators):
    """ Remove potential names from passed list of estimators """
    if len(estimators) == 0:
        return []
    else:
        if isinstance(estimators[0], tuple):
            estimators = [est for _, est in estimators]
        return estimators


def _check_names(estimators):
    """ Helper to ensure all estimators and transformers are named """
    # Check if empty
    if len(estimators) is 0:
        return []
    # Check if pipeline, if so split up
    elif isinstance(estimators, Pipeline):
        return estimators.steps
    # Check if named tuple
    elif isinstance(estimators[0], tuple):
        return estimators
    # Else assume list of unnamed estimators
    else:
        return _name_estimators(estimators)
